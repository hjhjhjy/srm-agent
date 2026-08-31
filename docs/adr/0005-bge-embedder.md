# ADR 0005：稠密向量嵌入器由 OfflineHashEmbedder 切换为 BGE

- 状态：**已接受**
- 日期：2026-08-31
- 关联：ADR 0001（编排）、ADR 0002（审批与幂等）、ADR 0003（记忆与网关）、ADR 0004（安全纵深）

## 背景

M2 各阶段（Phase 1~5）的检索后端 `HybridBackend` 一直使用 `OfflineHashEmbedder`
（字符二元切分 + 哈希抽桶的确定性向量）作为稠密召回信号。它在「零依赖、离线确定性、
CI 常绿」约束下表现良好，但与 BM25 经 RRF 融合后，稠密路只是「语义纠偏」——
对同义改写、跨词面语义的召回能力有限，无法发挥 v1 中 BGE 稠密向量的真正价值。

生产化要求语义召回质量上探，因此需要把默认稠密向量器替换为真实的 BGE 本地模型。
约束（与 M1/M2 一致，不可妥协）：

- **离线确定性**：未安装 `sentence-transformers` 时，必须自动回退到 `OfflineHashEmbedder`，
  保证本地开发 / CI / 单测永远可跑、且 100% 可复现。
- **零额外依赖（默认）**：`sentence-transformers`（及其拉取的 torch 等）不得进入 `requirements.txt`
  主依赖，避免 CI 安装巨型依赖；改为可选 `requirements-bge.txt`。
- **CI 常绿**：`pytest` / `ruff` / `mypy` 门禁全绿，测试不得触发真实模型下载。

## 决策

### 1. 实现真实的 `BGEEmbedder`，封装 BGE v1.5 的 query 指令前缀

`app/rag/embeddings.py` 中原先只有占位 stub，现补全为生产实现：

- **懒加载**：`from sentence_transformers import SentenceTransformer` 放在 `__init__` 内部，
  未装依赖时导入失败被外层 `try` 捕获并回退，不影响包导入与离线运行。
- **query 指令前缀**：BGE v1.5 系列在检索场景下对 query 需要加指令前缀
  （中文：`为这个句子生成表示以用于检索相关文章：`），文档侧**不加**。
  该细节封装在两个方法内 —— `embed(texts)` 编码文档、`embed_query(text)` 编码检索 query，
  调用方（检索编排层）无需关心前缀。非 v1.5 模型（如 `bge-m3`）默认不加前缀，
  可通过 `query_instruction` 参数显式覆盖 / 置空。
- **归一化**：`encode(normalize_embeddings=True)`，输出单位向量，使检索后端可直接用点积当余弦相似度。

### 2. 新增 `get_embedder()` 工厂，按环境变量选择并缓存

```python
# SRM_EMBEDDER=hash  强制 OfflineHashEmbedder（确定性、零依赖）
# SRM_EMBEDDER=bge   强制 BGEEmbedder（缺失依赖 / 模型即报错）
# 其他（默认 auto）  优先 BGEEmbedder，加载失败则回退 OfflineHashEmbedder 并告警
emb = get_embedder()   # 按 (mode, model) 缓存，避免重复加载大模型
```

`HybridBackend.__init__` 的默认 embedder 改为 `embedder or get_embedder()`，
`app/rag/seed.py` 的 `seed_kb()` 沿用 `HybridBackend()`，即生产启动时自动启用 BGE
（当环境装好依赖且有模型权重时），CI / 离线则静默回退 hash。调用方零改动。

### 3. 检索后端区分 query / document 编码路径

`hybrid.py` 的 `search()` 原先对 query 也调用 `embed([query])`；现改为
`embed_query(query)`，确保 BGE 的 query 前缀只在检索时生效，语料入库（`add()`）
仍走无前缀的 `embed(corpus)`。

### 4. 稠密路权重作为可调旋钮

`_dense_weight` 默认 0.35（针对 hash 语义噪声较大的保守值），现支持
`SRM_DENSE_WEIGHT` 环境变量上调（如 0.5~0.6），便于在启用 BGE 高质量稠密向量时
更充分利用语义召回，同时保留 BM25 作为中文关键词主信号。

### 5. 依赖隔离 + 测试分层

- `sentence-transformers` 移入可选 `requirements-bge.txt`，`requirements.txt` 仅留注释说明。
- 新增 `tests/test_embeddings.py`：
  - `OfflineHashEmbedder` 确定性、L2 归一、`embed_query == embed([t])[0]`；
  - `get_embedder` 的 hash 模式、auto 回退（用模拟异常覆盖 BGE 加载失败分支，不联网）、缓存；
  - BGE 的 query 前缀逻辑（注入假 `sentence_transformers`，无需真实模型）；
  - 真实 BGE 向量化（维度 / 形状 / 确定性 / 前缀差异）默认跳过，
    仅 `SRM_RUN_BGE_TESTS=1` 且本地有模型时运行，避免 CI 联网下载大模型。

## 后果

**正向**

- 生产环境语义召回质量上探至真实 BGE 水平，同义改写 / 跨词面检索显著改善。
- 离线 / CI 行为完全不变：无依赖时自动回退 hash，所有门禁常绿，回归风险为零。
- 切换是配置驱动的（`SRM_EMBEDDER` / `SRM_BGE_MODEL` / `SRM_DENSE_WEIGHT`），
  无需改代码，回滚只需改环境变量。

**代价 / 注意**

- 启用 BGE 需安装 `requirements-bge.txt`（引入 torch 等重依赖），且首次运行会从
  HuggingFace 下载模型权重（国内设 `HF_ENDPOINT=https://hf-mirror.com`）。
- 生产若用 BGE，建议把 `SRM_DENSE_WEIGHT` 上调并重新跑 `eval.py` 校准 top-K 召回指标；
  默认 0.35 对 BGE 偏保守，但仍是可用配置。
- 真实 BGE 路径的单测默认不跑，由本地 / 预发环境显式开启验证。
