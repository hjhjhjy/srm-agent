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

三级决策优先级，**拿不到人工决策时默认拒绝**（fail-closed）：

1. 持有 `approval:auto` scope → 策略自动放行（内部可信账号）
2. LangGraph `interrupt` 可用 → 挂起等待人工，恢复时记录审批人
3. 以上都不满足（无 interrupt 能力且非可信账号）→ **拒绝**

> 不存在「外部审批系统回调预设」这一独立分支：无 interrupt 能力时无法可靠挂起，
> 此时直接 fail-closed 比「放行」更安全。

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

### 7. M4：四层记忆 · 状态检查点 · 合规

Agent 的记忆从"一层对话缓冲"升级为**分层 + 可治理**的体系：

- **四层记忆**：工作（单次运行草稿）/ 情景（多轮对话，驱动指代消解）/ 语义（跨会话沉淀的事实与偏好）/ 程序（可复用流程模板）。`MemoryManager` 统一聚合，后端可插拔（默认内存，生产换 Redis/PG）。
- **状态检查点**：运行期每个节点边界都落一份完整状态快照，支持列出 / 读取 / 删除 / 回放，排查"跑到哪一步、当时状态是什么"不再靠猜。
- **合规闸**（默认开启）：写入记忆前自动 PII 脱敏（不落明文）、按层保留期自动清理、支持被遗忘权（`DELETE`）与数据导出（`GET`，DSAR）、所有合规操作可审计可观测。

```bash
# 导出某身份的记忆（数据可携）
curl http://localhost:8000/api/memory/identities/qlk/SUP001/export \
  -H "X-API-Key: <dev key>"

# 删除某身份的全部记忆（被遗忘权；跨身份需 compliance:manage）
curl -X DELETE http://localhost:8000/api/memory/identities/qlk/SUP001 \
  -H "X-API-Key: <dev key>"

# 查看某会话的检查点
curl http://localhost:8000/api/checkpoints/demo-1 -H "X-API-Key: <dev key>"
```

约束不变：离线确定性（默认 `ScriptedLLM` + 内存后端）、零额外依赖、CI 常绿。

## 快速开始

```bash
cd backend
pip install -r requirements.txt
pytest                      # 全部离线运行，无需 API Key
uvicorn app.main:app --reload
```

打开 http://localhost:8000/docs

鉴权（**凭据不再硬编码**，见 `app/core/config.py`）：

- 生产：设置 `SRM_JWT_SECRET`，用 `Authorization: Bearer <jwt>` 鉴权（JWT 由你的 IdP 签发）。
- 本地演示：设置环境变量 `SRM_DEV_API_KEY`（任意字符串），请求带 `X-API-Key`；
  若两者都未设置，启动期会自动生成一个**仅本进程有效**的临时 dev key 并打印到日志。
- 默认 dev 身份：租户 `qlk`、用户 `SUP001`、scopes `kb:read,order:read,ticket:write,calc:use,approval:review`。
- 审批回调 `/api/approvals/resume` **必须**携带 `approval:review` scope，否则 403。

```bash
# 知识问答
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: <你的dev key>" -H "Content-Type: application/json" \
  -d '{"question":"如何注册成为青山利康供应商？","session_id":"demo-1"}'

# 写操作 → 返回 pending_approval，等待人工审批
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: <你的dev key>" -H "Content-Type: application/json" \
  -d '{"question":"帮我建个工单，对账金额对不上","session_id":"demo-2"}'

# 人工批准（用 session_id，thread_id 由服务端派生；审批人取自身份，不信任客户端自填）
curl -X POST http://localhost:8000/api/approvals/resume \
  -H "X-API-Key: <你的dev key>" -H "Content-Type: application/json" \
  -d '{"session_id":"demo-2","approved":true}'
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
| 检索 | BGE + BM25 + RRF + 重排 | BGE 稠密 + BM25 + RRF 融合（默认 BGE，离线/CI 自动回退 hash） |
| 状态 | 活跃维护 | 新建，M2 已完成 BGE 接入 |

v1 的稠密检索已通过 `get_embedder()` 工厂接入真实 BGE（`app/rag/embeddings.py`），
离线/CI 无 `sentence_transformers` 时自动回退 `OfflineHashEmbedder`，**接口不变**——这正是抽象层的价值。

## 路线图

- [x] **M1** 状态图 + 工具中心 + HITL 审批 + 幂等 + 审计 + 护栏 + 租户隔离
- [x] **M2-Phase1** 治理收口：审批回调鉴权 · 凭据移出代码(JWT/环境变量) · 审计落盘+approver · 基础限流 · 知识库 seeding · 文档/代码矛盾消除
- [x] **M2-Phase2** 接入 v1 稠密检索（BGE 替换离线 hash 嵌入器）· 可观测性（OTel 链路追踪 + 成本归因 FinOps）· 评测门禁入 CI
- [x] **M3** 记忆与 LLM 硬化（Phase4）· 安全纵深 Prompt 注入/PII/外部隔离（Phase5）· BGE 嵌入器生产化
- [x] **M4** 四层记忆（工作/情景/语义/程序）· 状态检查点（逐节点快照/回放）· 合规策略（PII 脱敏/保留期/被遗忘权/DSAR）
- [ ] **M5** Helm + CI/CD + 金丝雀 + SLO
- [ ] **M6** MCP 协议 · Prompt A/B · 知识版本化

## 文档

- [需求对齐文档](docs/00-requirements-alignment.md)
- [ADR 0001：选用 LangGraph 状态图编排](docs/adr/0001-langgraph-orchestration.md)
- [ADR 0002：写操作的审批与幂等设计](docs/adr/0002-hitl-approval-and-idempotency.md)
- [ADR 0003：记忆模块与 LLM 网关硬化](docs/adr/0003-memory-and-llm-hardening.md)
- [ADR 0004：安全纵深（注入/PII/隔离）](docs/adr/0004-security-hardening.md)
- [ADR 0005：BGE 稠密嵌入器生产化](docs/adr/0005-bge-embedder.md)
- [ADR 0006：成本归因 FinOps](docs/adr/0006-cost-attribution.md)
- [ADR 0007：四层记忆 · 状态检查点 · 合规策略](docs/adr/0007-four-layer-memory-checkpoint-compliance.md)
