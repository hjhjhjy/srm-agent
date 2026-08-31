# ADR 0003：记忆模块与 LLM 网关硬化

- 状态：**已接受**
- 日期：2026-08-31
- 关联：ADR 0001（编排）、ADR 0002（审批与幂等）

## 背景

M2 Phase 3 已交付可观测性，但编排链路存在三处"骨架已立、接线未通"的缺口：

1. **多轮指代失效**：`router/planner/responder` 三个节点各自独立构造 LLM 提示，**不会**携带历史对话。用户第二轮问"它具体需要准备哪些材料？"中的"它"无法消解，回答会凭空猜测或检索失败。
2. **Token 计量只在 responder 记一笔**：`budget.consume_tokens` 仅在应答节点调用一次，路由、规划、反思三个"吃 token"的 LLM 调用完全游离于预算护栏之外，护栏形同虚设。
3. **LLM 网关降级链单薄**：`OpenAICompatLLM` 只有"主模型 → 抛错"两档，没有备用模型档，也没有退避重试；瞬时网络抖动会直接击穿到编排层异常。

约束（与 M1/M2 一致，不可妥协）：

- **离线确定性**：默认 `ScriptedLLM`，CI 注入后全程规则降级，无 flaky test。
- **零额外依赖**：记忆/计量/降级均为纯 Python，不引入任何新三方包。
- **CI 常绿**：`pytest` / `ruff` / `eval` 门禁全绿。

## 决策

### 1. 进程级会话记忆单例 + 编排层读写

新增 `app/agent/session.py`：模块级 `MemoryStore` 单例，提供
`get_memory_store()` / `reset_memory_store()`。不依赖任何数据库，
天然满足"零额外依赖 + 单进程确定性"。

`run_agent` 在调用编排前先从 store 取该 session 的上下文注入
`dialogue_context` 字段，调用后再把本轮问答 `append` 回 store：

```python
store = get_memory_store()
ctx = store.context(session_id)            # 取历史摘要 + 近期明细
state = initial_state(question, ..., dialogue_context=ctx)
result = await graph.ainvoke(state, config=config)
store.append(session_id, "user", question)
store.append(session_id, "assistant", result.get("answer") or "")
```

`MemoryStore` 的折叠策略：`recent` 超过 `max_recent=6` 时，较早的一半
折叠进 `summary`（每条截取前 200 字），保证长会话下注入长度有界。
`dialogue_context` 由 `memory.build_coref_context` 提炼为"近期对话"片段，
供节点拼接到 user 提示中。

### 2. 三节点注入 `dialogue_context` 消解指代

`router`、`planner`、`responder` 构造 user 提示时统一改为：

```python
user_content = f"{ctx}\n\n用户问题：{q}" if ctx else q
```

`ctx` 即注入的 `dialogue_context`。这样"它/这个/上述"类指代，在路由分诊、
规划工具选择、最终应答三个阶段都能看到前文，回答不跑偏。
`has_coref` 提供轻量指代检测（零依赖词表），供后续按需扩展。

### 3. Token 计量贯穿全链路

`gateway.BaseLLM.achat` 与 `OpenAICompatLLM.achat` 新增 `phase` 参数，
透传给 `metrics.record_llm(..., phase=phase)` 做成本归因。
`router / planner / reflector / responder` 四个 LLM 调用点**均**在返回前
调用 `budget.consume_tokens(resp.tokens)` 并回写 `budget` 键，由 LangGraph
状态归约聚合，护栏（step/token/循环检测）从"只看 responder"升级为"全节点可见"。

### 4. LLM 网关降级链硬化

`OpenAICompatLLM` 升级为三档：

1. **主模型**：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`，经
   `_call_with_retry` 做指数退避重试（`max_retries=2`，`retry_base=1.0`，
   退避 = `retry_base * 2**attempt`，单调用超时 25s 低于 30s 墙钟预算）。
2. **备用模型**：主模型失败时若配置了 `LLM_*_SECONDARY`，自动切到备用档重试。
3. **规则兜底**：备用仍失败则抛错，编排层因 `ScriptedLLM`（离线）或空 content
   自然降级为"检索直答"——不会出现硬 500，符合 fail-soft 原则。

新增模块级 `_retry_async`（async 指数退避通用封装，带异常类型白名单），
与主调用链路解耦，便于复用与测试。

## 理由

- **记忆接线**是"对话式 Agent"与"单轮问答脚本"的分水岭；不接记忆，多轮业务咨询（"供应商注册"→"它要哪些材料"）无法成立。
- **计量贯通**是预算护栏可信的前提；只记应答节点的 token，等于给护栏蒙了眼。
- **降级链硬化**是真实网络环境下"不因一次抖动整链路雪崩"的工程底线；备用模型 + 退避把瞬时故障吸收在网关内。

## 影响

- 记忆单例为**进程级**，多副本部署下不同进程不共享会话（与 ADR 0002 指出的 `MemorySaver` 限制同源）。M4 多副本需替换为外部会话存储（Redis/Postgres），当前单进程内确定性正确。
- `phase` 标签使 Prometheus 指标可按 `router/planner/reflector/responder` 拆分成本，便于定位哪段提示最费 token。
- 离线 CI 下 `ScriptedLLM` 返回 `tokens=0`，`budget.tokens_used` 恒为 0，护栏验证逻辑不依赖真实计量；真实模型接入后计量自动生效。
- `reset_memory_store()` 已接入 `tests/conftest.py` 的 autouse fixture，保证每个用例记忆隔离、测试确定性。

## 验证

- `pytest -q`：**43 passed**（含新增 `test_memory.py` 8 例、`test_phase4.py` 4 例）。
- `ruff check app tests scripts`：全绿。
- `scripts/demo.py` 场景 4：第 1 轮问"如何注册成为青山利康供应商？"，第 2 轮问"它具体需要准备哪些材料？"——注入的 `dialogue_context` 含首轮问题，指代消解生效。
- `eval` 门禁维持 95/100/95/100（记忆/计量改动不影响检索与回答质量）。
