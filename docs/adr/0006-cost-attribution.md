# ADR-0006：LLM 成本归因（FinOps）

- 状态：**已采纳**（2026-08-31）
- 领域：可观测性 / 横切面 B（护栏与可观测）
- 关联：ADR-0003（记忆与 LLM 硬化）、ADR-0005（BGE 嵌入器）

## 背景（Context）

需求对齐文档（§7、§10 #1）把「LLM 成本治理」列为"能跑的 Demo"与"敢上生产"的分水岭，
企业客户最先追问的就是「一个月烧多少钱」。M3 的可观测性此前已落地：

- 链路追踪（span tree + 可选 OTel 桥接）；
- Prometheus 指标：`srm_llm_tokens_total`（按 model/phase）、`srm_llm_calls_total`、`srm_llm_duration_seconds`；
- `.github/workflows/ci.yml` 中的离线评测门禁。

但缺失一环：**token 数 ≠ 钱**。现有指标只能回答"消耗了多少 token"，无法回答
"哪个租户、哪个模型花了多少美元"，也就无法做租户级配额、预算熔断、成本异常告警。

约束（与全项目一致）：

- **离线确定性、零额外依赖、CI 常绿**：不能引入计费 SDK 或外部账单系统；
- 模型定价是**展示用估算**，不是计费系统，不能对外承诺精确到分；
- 成本必须能按 `tenant` 归因，且归因逻辑不能靠逐层透传参数污染节点签名。

## 决策（Decision）

1. **在 `observability/metrics.py` 增加 `LLM_COST` Counter（标签 `tenant`/`model`）**，单位美元。
   由新增的 `record_llm_cost(tenant, model, prompt_tokens, completion_tokens) -> float` 累加，
   返回本次成本便于回显。`cost_summary()` 提供按 `tenant/model` 聚合的读数（已排除 prometheus 的 `_created` 时间戳样本）。

2. **定价表内置于 `metrics.py`**（`_DEFAULT_PRICING`，美元 / 1K token，区分 in/out 价），
   覆盖 deepseek / gpt-4o* / qwen 等常见模型，未知模型回退 `default`。
   允许环境变量 `SRM_MODEL_PRICING`（JSON）覆盖，便于随行就市更新而**不改动代码**。

3. **扩展 `record_llm` 签名**：新增 `prompt_tokens` / `completion_tokens` 可选参数。
   有 token 拆分时调用 `record_llm_cost(get_tenant(), model, ...)` —— tenant 取自上下文变量，
   默认 `"unknown"`。无拆分（错误路径、0 token）则不产生成本，避免污染指标。

4. **租户身份用 `contextvars` 透传，不污染节点签名**：在 `observability/tracing.py`
   新增 `tenant_var` / `user_var` 及 `set_identity()` / `get_tenant()` / `get_user()` /
   `reset_identity()`。请求入口（`main.py` 的 `/api/chat`、`/api/chat/stream`、`/api/approvals/resume`）
   在鉴权拿到 `Identity` 后调用 `set_identity(tenant_id, user_id)`。LLM 网关、工具、节点
   均可在不修改函数签名的前提下读到当前租户，实现成本与审计的自动归因。

5. **`LLMResponse` 增加 `prompt_tokens` / `completion_tokens` 字段**：
   - `OpenAICompatLLM` 直接从 `usage.prompt_tokens` / `usage.completion_tokens` 取值；
   - `ScriptedLLM`（离线/CI）按消息体积做确定性拆分（输入≈全部消息长度，输出≈生成长度），
     保证离线环境也有可复现的成本估算，且不依赖真实用量。

## 后果（Consequences）

**正面**

- 现在能回答"每个租户、每个模型花了多少钱"，支撑租户级配额、预算熔断、成本异常告警；
- 归因零参数透传：新增节点/工具不需为成本改动签名；
- 定价可经环境变量热更新，定价变化不触发代码改动与重新发布；
- 离线 CI 同样产生成本指标（ScriptedLLM 估算），门禁一致性好；
- 成本与既有 token/调用数/耗时指标同处 Prometheus，可直接进同一块 Grafana 面板。

**负面 / 注意**

- 这是**成本估算**而非真实账单：依赖公开定价与 token 拆分估算，存在偏差，不可直接用于对客计费；
- 成本精度受限于 `prompt/completion` 拆分的准确性（真实 API 提供精确 usage；离线为估算）；
- `LLM_COST` 为单调递增 Counter，长期运行需配合 Prometheus 的 `rate()` / `increase()` 看增量，
  绝对累计值会持续增大（属预期）。

## 替代方案（被否决）

- *引入计费 SDK / 云厂商成本 API*：违反"零额外依赖、离线确定性"，且把成本口径绑死在单一云厂商。
- *在 `record_llm` 强制要求调用方传 tenant*：会污染 `gateway`/`nodes` 全部调用签名，
  与"节点签名稳定、LangGraph 可正常调用"的既有约束冲突。
- *只在 `main.py` 按请求级聚合成本*：无法分摊到单次 LLM 调用（router/planner/reflector/responder
  各阶段），也就无法做"哪个环节在烧钱"的定位。
