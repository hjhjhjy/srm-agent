# ADR 0007：四层记忆 · 状态检查点 · 合规策略

- 状态：**已接受**
- 日期：2026-08-31
- 关联：ADR 0003（记忆与 LLM 硬化）、ADR 0004（安全纵深）、ADR 0005（BGE 嵌入器）、ADR 0006（成本归因）

## 背景

Phase 4 交付了**一层**对话记忆（`MemoryStore` + `dialogue_context` 注入），解决了多轮指代。
但企业级 Agent 的记忆应是分层的，且必须满足合规要求。当前缺口：

1. **记忆只有一层**：对话缓冲既当工作草稿、又当长期知识、又当用户画像，职责混淆，
   无法跨会话沉淀"这个供应商对账周期 30 天"之类的可复用知识。
2. **没有显式检查点**：LangGraph 的 `MemorySaver` 与图实例绑定、不可直接 inspect/导出，
   排查"跑到哪一步、当时状态是什么"困难，且无法做审计式轨迹留存与回放。
3. **记忆不合规**：对话内容（可能含手机号/邮箱）被原样存入进程内存，无 PII 脱敏、
   无保留期、无被遗忘权、无数据导出——GDPR/个保法的"存储限制""被遗忘权""可携权"全缺。

约束（与 M1~M3 一致，不可妥协）：

- **离线确定性**：默认 `ScriptedLLM` + 内存后端，CI 注入后全程规则降级，无 flaky test。
- **零额外依赖**：记忆/检查点/合规均为纯 Python（stdlib + 既有 langchain/pydantic），不引入任何新三方包。
- **CI 常绿**：`pytest` / `ruff` / `mypy` / `eval` 门禁全绿。

## 决策

### 1. 四层记忆模型（`app/agent/memory_layers.py`）

| 层 | 含义 | 默认后端 | 生命周期 |
|---|---|---|---|
| **Working 工作** | 单次运行的计划/工具留痕/反思/迭代 | 不持久化（从 `AgentState` 抽取） | 随请求生灭 |
| **Episodic 情景** | (租户,用户,会话) 的对话轮次 | `InMemoryEpisodicBackend` | 30 天（保留期） |
| **Semantic 语义** | key→事实/偏好/已解决问答对 | `InMemorySemanticBackend` | 180 天 |
| **Procedural 程序** | 命名可复用流程模板（步骤序列） | `InMemoryProceduralBackend` | 365 天 |

- 各后端均为接口 + 内存实现，生产可替换为 Redis/Postgres，对外接口不变。
- `MemoryManager` 聚合四层，向编排层/API 暴露统一方法（`record_turn` / `add_fact` /
  `search_facts` / `save_procedure` / `forget_identity` / `export_identity` / `sweep`）。
- 情景层的 `recent_context` 复用 Phase 4 的 `build_coref_context`，保证多轮指代行为一致。
- 语义检索用零依赖的字符级分词（英文按词、中文单字 + 二元组），不引入 jieba。

### 2. 显式状态检查点（`app/agent/checkpoint.py`）

- `snapshot_state` / `restore_state`：把 `AgentState`（含 pydantic 模型与 langchain 消息）
  序列化为 JSON 友好 dict 并反向重建，互为逆操作，保证快照可往返。
- `CheckpointStore`（接口）+ `InMemoryCheckpointStore`：按 `(thread_id, node)` 命名存快照，
  支持 `save / load / list / delete / delete_thread`。
- 编排集成：``run_agent`` 新增可选 `checkpoint_store` 参数；传入后改用
  ``graph.astream(stream_mode="updates")`` 逐节点推进，并在每个节点后通过
  ``graph.aget_state`` 抓取完整状态落快照，覆盖 router→planner→executor→reflector→responder。
- `resume_from_checkpoint`：从指定检查点重建状态并以独立 replay 线程重新执行，支撑
  审计/回放（避免与原始线程在 `MemorySaver` 中的状态相互合并，语义确定）。

### 3. 合规记忆层（`app/agent/compliance.py`）

`ComplianceManager(MemoryManager)` 在**写入与清理**时统一施加合规闸：

- **PII 不落明文**：写入情景/语义/程序记忆前对内容做 PII 脱敏（复用 Phase 5 `mask_pii`），
  原文仅留指纹 hash，绝不存储明文敏感信息。
- **保留期（存储限制）**：各层不同 TTL，`sweep` 按当前时间清理过期数据。
- **被遗忘权**：`forget_identity` 删除某身份在全部分层的全部数据并打点。
- **数据可携（DSAR）**：`export_identity` 返回该身份全部（已脱敏）记忆。
- **访问审计**：删除/导出/清理均追加 `compliance_audit` 并打 Prometheus 指标
  （`srm_compliance_*`），使合规可观测、可问责。

### 4. API 暴露（`app/main.py`）

- `POST /api/chat` 经 `run_agent` 统一入口，自动写入四层合规记忆 + 逐节点检查点。
- `DELETE /api/memory/identities/{tenant}/{user}` 被遗忘权（本人或 `compliance:manage`）。
- `GET    /api/memory/identities/{tenant}/{user}/export` 数据导出（DSAR）。
- `GET    /api/checkpoints/{session_id}` 列出某会话全部检查点。
- `DELETE /api/checkpoints/{session_id}/{checkpoint_id}` 删除单检查点。

鉴权：记忆/检查点操作仅本人可操作自身数据，跨身份操作需 `compliance:manage` scope。

## 理由

- **分层**是"对话式 Agent"与"有状态的长期助手"的分水岭；不分层，知识/画像无法跨会话沉淀。
- **显式检查点**把"跑到哪了"变成可查询、可回放、可审计的一等公民，而非黑盒里的内部状态。
- **合规前置**比事后补救便宜得多：PII 在落盘前脱敏、保留期自动清理，使"合规"成为架构约束
  而非各业务代码里散落的 if。

## 影响

- 新增 `memory_layers.py` / `checkpoint.py` / `compliance.py` 三个模块，外加 metrics 合规指标。
- `run_agent` 增加 `memory_manager` / `checkpoint_store` / `thread_id` 可选参数，默认行为不变
  （无参数时与 Phase 4 完全一致，demo 脚本向后兼容）。
- `conftest` autouse fixture 新增 `reset_memory_manager` / `reset_checkpoint_store`，保证测试隔离。
- 进程级单例 `get_memory_manager()` 默认返回 `ComplianceManager`；生产替换为外部后端时仅改这一处。
- 与 Phase 4 `session.MemoryStore` 并存：前者驱动多轮 `dialogue_context`（已验证），后者为
  合规长久记录 + 语义/程序层 + API。生产收敛方向是统一到合规 `MemoryManager`。

## 验证

- `pytest`：新增 `test_m4_memory.py` / `test_m4_checkpoint.py` / `test_m4_compliance.py`
  共 23 例（四层 CRUD、租户隔离、快照往返、逐节点检查点、检查点回放、PII 脱敏、被遗忘权、
  DSAR 导出、保留期清理、API 端到端），**全量 ~99 例通过**。
- `ruff check app tests scripts`：全绿。
- `mypy app`：no issues。
- `scripts/demo.py` 场景 4 多轮指代行为不变（Phase 4 路径未改）。
