# SRM 企业级 Agent

> 面向供应商的企业级 Agent 应用：**不只是能答，而是能办事、可审计、可回滚、成本可控**。
>
> 基于已有 RAG 项目（v1 `srm-rag`）升级而来，独立仓库，不改动原项目。

## 为什么不是"又一个 RAG Demo"

v1 解决的是**检索质量**（混合检索 + 重排，检索准确率 90%）。但企业级落地的真正难点在检索之外：

| 问题 | v1 | 本项目 |
|---|---|---|
| 能查实时业务数据吗？ | 不能，只能答静态知识 | ✅ 工具调用（订单/发票/对账/工单） |
| 多步任务怎么办？ | 固定管线，无规划 | ✅ LangGraph 状态图 + 反思迭代 |
| Agent 失控了怎么办？ | 无护栏 | ✅ 步数/token/墙钟三重预算 + 死循环检测 |
| 写操作谁批准？ | 无写操作 | ✅ HITL 审批门 + 幂等键 + 审计留痕 |
| 多租户会串数据吗？ | ⚠️ 检索无租户过滤 | ✅ 检索层 + 工具层双重行级隔离 |
| 出问题能定位吗？ | 只有汇总指标 | ✅ 执行轨迹可回放（M3 接入 OTel） |
| 一个月烧多少钱？ | 不知道 | ✅ token 计量埋点（M3 成本归因） |

## 架构

```
START → router ─┬─────────────────────────────→ responder → END
                └→ planner → executor ─┬→ approval ─┬→ executor（批准后执行写操作）
                                       │            └→ responder（拒绝/未决）
                                       └→ reflector ─┬→ planner（信息不足，重规划）
                                                     └→ responder（充分或护栏触发）
```

六层架构 + 两道横切面：

```
接入层 Gateway   REST · SSE 流式 · 多租户路由
编排层 Orchestrator  LangGraph 状态图 · 规划 · 工具调度 · 迭代护栏
能力层 Tools     工具注册中心 · 知识检索 · SRM 业务 API · 工单
认知层 Cognition  多模型路由 · 结构化输出 · 可注入 mock
记忆层 Memory    工作 · 会话 · 长期画像 · 知识库（M4）
存储层 Storage   pgvector · PostgreSQL · Redis（M2/M4）
    ⇅ 横切面：安全与权限（工具级授权 · 审计 · 隔离） / 护栏与可观测（注入防护 · 追踪 · 成本）
```

## 核心能力

### 1. 工具级授权（不是 API 级）

权限校验下沉到工具本身。LLM **只看得到**调用方有权调用的工具，执行前再校验一次：

```python
@registry.tool(
    description="创建人工工单（写操作，需审批）",
    args_model=TicketCreateArgs,
    required_scopes=("ticket:write",),
    side_effect=True,   # → 触发 HITL 审批门
    idempotent=True,
)
```

只读用户调用写工具 → 工具不在可见列表 → 即使幻觉出来也在执行前被拦截并记入审计。

### 2. HITL 审批门（写操作必须过人）

四级决策优先级，**拿不到人工决策时默认拒绝**（fail-closed）：

1. 持有 `approval:auto` scope → 策略自动放行（内部可信账号）
2. LangGraph `interrupt` → 挂起等待人工
3. 状态预设 → 外部审批系统回调
4. 以上都不满足 → **拒绝**

### 3. 幂等性（写操作的生命线）

相同 `idempotency_key` 直接回放首次结果。网络重试、用户重复点击都不会重复建单：

```python
first  = await invoke("ticket_create", args, ctx(idem="key-abc"))  # replayed=False
second = await invoke("ticket_create", args, ctx(idem="key-abc"))  # replayed=True，同一工单号
```

缺幂等键的写操作**直接拒绝执行**——这是硬约束，不是建议。

### 4. 护栏（防止 Agent 烧钱跑飞）

| 护栏 | 默认阈值 | 超限行为 |
|---|---|---|
| 步数上限 | 6 步 | 强制收敛并标注"未完全解决" |
| Token 预算 | 8000 / 请求 | 触发压缩，再超降级 |
| 墙钟超时 | 30s | 中断并返回已得结果 |
| 死循环检测 | 同参数调用 > 3 次 | 强制跳出 |

**护栏触发后仍会给出答案**（优雅降级），而不是抛异常。

### 5. 租户隔离（补 v1 的 P0 缺口）

v1 的检索只有 `flow_code` 过滤，多租户下会串数据。本项目的检索后端**强制**过滤：

```python
if hit.tenant_id not in ("public", tenant_id):
    continue
```

工具层再做一次行级过滤，双重保险。

### 6. 计算器不用 eval

工具参数是 LLM 生成的，而 LLM 输出可能被检索内容里的 Prompt 注入污染。用 `eval` 等于开 RCE 口子。
这里用 **AST 白名单求值**：只允许数字常量与四则运算，禁止变量引用、属性访问、函数调用。

## 快速开始

```bash
cd backend
pip install -r requirements.txt
pytest                      # 全部离线运行，无需 API Key
uvicorn app.main:app --reload
```

打开 http://localhost:8000/docs

演示凭据（M1 静态 Key，M2 替换为 JWT）：

| Key | 权限 |
|---|---|
| `dev-supplier-key` | 知识检索 + 订单查询 + 建工单（需审批） |
| `dev-readonly-key` | 仅只读，看不到写工具 |
| `dev-admin-key` | 全部权限 + `approval:auto`（工单免审批） |

```bash
# 知识问答
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: dev-supplier-key" -H "Content-Type: application/json" \
  -d '{"question":"如何注册成为青山利康供应商？","session_id":"demo-1"}'

# 写操作 → 返回 pending_approval，等待人工审批
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: dev-supplier-key" -H "Content-Type: application/json" \
  -d '{"question":"帮我建个工单，对账金额对不上","session_id":"demo-2"}'

# 人工批准
curl -X POST http://localhost:8000/api/approvals/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-2","approved":true,"reviewer":"buyer01"}'
```

## 测试

```bash
pytest -v
```

测试全部使用 `ScriptedLLM`（脚本化响应）与内存知识库，**确定性可复现，不需要 API Key，不联网**。
覆盖：只读链路、写操作审批通过、审批拒绝、越权拦截、幂等、租户隔离、护栏触发、注入防护。

## 与 v1 的关系

| | v1 `srm-rag` | v2 `srm-agent` |
|---|---|---|
| 定位 | 检索质量深度 | 工程完备度 |
| 检索 | BGE + BM25 + RRF + 重排 | M1 内存 BM25，**接口预留** M2 接入 v1 稠密检索 |
| 状态 | 活跃维护 | 新建，M1 进行中 |

v1 的稠密检索将在 M2 通过 `set_backend()` 接入，**接口不变**——这正是抽象层的价值。

## 路线图

- [x] **M1** 状态图 + 工具中心 + HITL 审批 + 幂等 + 审计 + 护栏 + 租户隔离
- [ ] **M2** 接入 v1 稠密检索 · 工具级授权下沉 · 密钥改造
- [ ] **M3** OpenTelemetry 链路追踪 · 成本归因 · 评测门禁入 CI
- [ ] **M4** 四层记忆 · 状态检查点 · 合规策略
- [ ] **M5** Helm + CI/CD + 金丝雀 + SLO
- [ ] **M6** MCP 协议 · Prompt A/B · 知识版本化

## 文档

- [需求对齐文档](docs/00-requirements-alignment.md)
- [ADR 0001：选用 LangGraph 状态图编排](docs/adr/0001-langgraph-orchestration.md)
- [ADR 0002：写操作的审批与幂等设计](docs/adr/0002-hitl-approval-and-idempotency.md)
