# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 常用命令

```bash
# 交互式命令行
python -m agent_pipeline.main

# Web 前端（推荐）
python -m uvicorn web_app.app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000，直接查看检索图像

# 运行批量测试（25 个预定义查询）
python -m test_queries.main

# 下载 CLIP 模型
python scripts/download_models.py --mirror hf-mirror

# 拉取 Ollama 模型（默认）
ollama pull qwen3:8b           # 意图识别
ollama pull qwen3-vl:8b        # VL_Refine 细粒度重排序

# 旧版模型（已弃用，仅作参考）
ollama pull qwen2.5:3b        # 意图识别（旧）
ollama pull qwen2.5vl:3b      # VL_Refine（旧）
```

项目依赖通过 conda 环境 `MmIRS` 管理（含 `fastapi`、`uvicorn` 等 Web 依赖）。运行任何 Python 命令前需激活该环境或使用其完整路径：

```bash
conda activate MmIRS && python -m agent_pipeline.main
# 或
C:\Users\21440\.conda\envs\MmIRS\python.exe -m agent_pipeline.main
```

# 运行意图识别实验（独立，不影响主程序）
conda activate MmIRS && python -m experiment.intent_experiment.run_intent_experiment

项目未配置代码检查或单元测试框架。`test_queries/` 是手动集成测试运行器。

**实验隔离原则：** `experiment/` 目录下的脚本和数据仅做评估用途，只读取 `intent_module/` 等核心模块进行推理，不修改任何主程序代码。新增实验时遵循同样原则。

## 架构

系统采用 **程序化路由编排**（对应 v1.md 架构图），接收自然语言图像查询，通过本地 LLM 判断意图，然后由代码做单一决策：是否需要 VL 精排。

```
用户查询 → 意图识别(LLM) → needs_vl? (计数 或 复杂属性)
                                ├── 是 → VL精排(CLIP粗排+VL验证计数/属性)
                                └── 否 → 常规检索(CLIP全短语编码) → 回复(模板)
```

路由是**代码决策**（`pipeline.py` 中 `needs_vl = bool(object_count) or bool(complex_attrs)`），不由 LLM 自行选择。LLM 负责 2 个核心环节：意图识别、VL 精排。回复使用纯模板。

### Web 前端：`web_app/`

`web_app/app.py` 为 FastAPI 应用，在启动时（lifespan）创建 `MultiModalAgentPipeline` 单例。核心端点：

- `POST /api/chat/stream` — SSE 流式检索：推送进度事件（intent → retrieval → vl_refine → formatting）→ 最终返回结构化 JSON（含图片路径、各阶段分数、路由信息）
- `GET /api/health` — 服务健康检查
- `/images/` — 从 `test_images/` 静态提供图片文件
- `/` — 单文件前端（`static/index.html`），零外部依赖

前端通过 SSE 接收进度（VL 精排阶段实时显示当前候选图像名和进度条），结果以缩略图网格展示，点击打开灯箱查看大图和分数详情。Pipeline 中 `chat()` 和 `chat_structured()` 共享核心逻辑 `_chat_impl()`（仅返回格式不同），`chat_structured()` 额外支持 `progress_callback` 回调，将 VL 精排的逐候选进度实时推送到前端。

### 编排器：`agent_pipeline/pipeline.py`

`MultiModalAgentPipeline` 将所有组件串联：
1. 创建 `IntentRecognitionModule`（通过 Ollama 运行 LLM，模型由 `INTENT_MODEL` 环境变量控制）。
2. 创建共享的 `CLIPEncoder`（`shared/clip_encoder.py`），从 `models/clip_ViT` 加载 `openai/clip-vit-large-patch14`。
3. 创建 `RegularRetrievalModule`，传入共享的 CLIP 编码器。
4. 创建 `FineGrainedRetrievalModule`，同时传入共享的 CLIP 编码器 **和** 常规检索模块的 `OfflineIndexer`——使两个模块查询同一个向量数据库。
5. 通过 `data_load.py` 发现图像，经 `RegularRetrievalModule.offline_indexing()` 建立索引。
6. `chat()` 方法：意图识别 → 判定路由（代码决策）→ 检索 → 模板格式化回复。

### 路由逻辑（`pipeline.py:chat()` 方法）

- 单一决策：`needs_vl`（物体数量 或 非CLIP友好属性）
- `IntentRecognitionModule.analyze_intent()` 解析用户查询。
- 如果 `need_retrieval=False` → 直接回复"不需要检索"。
- 如果未提取到类别 → 回退：用原始查询做常规检索。
- 如果 `needs_vl` → **VL 精排**：CLIP 粗排（裸类别）→ VL 验证（计数 / 属性合并为一次调用）。
- 如果 `!needs_vl` → **CLIP-only**：编码全短语（如 `"白色的狗"`、`"3只狗"`）直接检索。

CLIP 友好属性（颜色/大小/明暗）与无属性查询归入同一条 CLIP-only 路径。计数与复杂属性（姿态/场景/视角等）归入同一条 VL 路径。

### 意图识别：`intent_module/`

`IntentRecognitionModule` 将用户查询通过 LangChain 链 `PromptTemplate` → `ChatOllama` → `StrOutputParser` 发送。结构化输出由 `parser.py` 解析为：`need_retrieval`、`category`、`count`、`count_type`、`method`（统一 TopK）、`attributes`、`object_count`。

`parser.py` 包含多层回退逻辑：LLM 未提取到类别时从常见物体列表匹配；LLM 误判数量时从原始 query 提取数字覆盖。`object_count`（图片里有几个物体）与 `count`（返回几张图片）通过量词区分——`张/幅` 是图片量词，`只/个/条/头/匹/群` 是物体量词，物体量词统一用 `个` 构建 VL 提示词。

### 常规检索：`regular_retrieval_module/`

针对**无属性条件查询**的单阶段 CLIP 检索：
- `OfflineIndexer` 通过 CLIP 编码所有图像，在内存中存储 `{url: {image_feature, raw_image}}`。编码完成后将特征向量持久化到 `cache/image_features.pt`，后续启动若图像列表未变则直接加载缓存跳过重编码。
- `Retriever` 将中文查询直接通过 Chinese RoBERTa 编码，计算与所有已索引图像的点积相似度，返回 Top-K 或阈值过滤结果。

### 细粒度检索：`fine_grained_retrieval_module/`

针对**需要 VL 精排的查询**（物体数量 或 复杂属性），采用两阶段检索：

1. **粗排阶段**（`OnlineRetriever`）：只用裸类别构建查询，经 Chinese RoBERTa 编码，在共享向量数据库中做相似度搜索。有物体数量时 candidate pool = `count × object_count × 2`（保底 30），否则取 `count × 3`（保底 15）。
2. **精排阶段**（`VLRefiner.refine()`）：所有粗排候选送 VL 做二分类验证。如有物体数量则验证计数（"恰好有 N 个 {类别}"），如有属性则合并验证（"恰好有 N 个 {属性}的{类别}"）。通过则保留 CLIP 粗排分，不通过则剔除。默认串行验证，可通过 `VL_PARALLEL_WORKERS` 环境变量开启并行（需配合 `OLLAMA_NUM_PARALLEL`，详见 README 已知限制）。每个 worker 持有独立的 `ChatOllama` 实例避免线程竞争。精排后裁剪到图片请求数量。

### 已知瓶颈

**VL 精排精度仍不及 SOTA API（如 Gemini-Flash、GPT-4o）**，复杂属性组合（"棕色站立的狗"）的细粒度判断仍存在漏判。理想混合架构（CLIP 本地粗筛 + 云端多模态 API 精排）详见 README「架构演进」章节。

**不支持专名/人名检索**。CLIP 文本编码器对专有名词（人名、地名、品牌等）无感知——这些 token 在训练语料中极少出现或不存在，无法编码为有意义的 embedding 向量。当用户查询包含专名时，意图模块会将其退化为泛化类别（"庄方宜"→"人"），检索本质是在用泛化类别做相似度搜索。由于 Taiyi-CLIP 的零样本泛化能力，即便图像集中没有对应类别，模型也会在 embedding 空间中找出距离最近的图像（如将"人"映射到视觉最接近的灵长类动物），但结果与原始专名无实际关联。同时 **TopK 路径不设最低相似度阈值**，导致低分匹配照常返回，产生虚假的相关性印象。如需支持专名检索，需引入额外的人脸标注、图像标签或关键词索引等信号源。

### 共享组件：`shared/`

- `CLIPEncoder`：从本地路径加载 `openai/clip-vit-large-patch14`。提供 `encode_text()`、`encode_images()` 和 `get_logit_scale()`。

## 配置项

| 配置项 | 环境变量 / 位置 | 默认值 |
|------|------|---------|
| **意图识别 LLM** | `INTENT_MODEL` | `qwen3:8b` |
| **VL 模型** | `VL_MODEL` | `qwen3-vl:8b` |
| **VL 并行 worker 数** | `VL_PARALLEL_WORKERS` | `1`（串行；并行需配合 `OLLAMA_NUM_PARALLEL`） |
| **Web 服务端口** | `uvicorn --port` 参数 | `8000` |
| **Web 服务主机** | `uvicorn --host` 参数 | `0.0.0.0` |
| CLIP 模型路径 | `*/constants.py`（基于 PROJECT_ROOT 计算） | `models/clip_ViT` |
| 索引规模 | `REGULAR_INDEX_SIZE` | `-1`（全部图像） |
| HuggingFace 镜像 | `pipeline.py:8` / `HF_ENDPOINT` | `https://huggingface.co` |
| VL 精排策略 | `fine_grained_retrieval_module/vl_models.py` | 二分类（是/否），通过则保留 CLIP 粗排分 |
| 粗排候选倍数 | `agent_pipeline/pipeline.py` | count × 3（保底 15） |
| 磁盘缓存 | `regular_retrieval_module/offline_indexer.py` | `cache/image_features.pt` |
| Top-K 默认值 | `pipeline.py:_resolve_top_k()` | 5 |

## 关键设计模式

- **程序化路由**：路由决策由代码（`needs_vl = bool(object_count) or bool(complex_attrs)`）而非 LLM 做出，消除小模型路由不可靠的问题。
- **共享向量数据库**：`RegularRetrievalModule` 拥有 `OfflineIndexer`；`FineGrainedRetrievalModule` 通过构造函数注入接收。两个模块共享同一个 `CLIPEncoder` 实例。
- **Taiyi-CLIP 双模型编码**：文本编码使用 Chinese RoBERTa（`models/Chinese_RoBERTa/`），原生支持中文；图像编码使用 CLIP ViT-L/14（`models/clip_ViT/`）。两者经投影对齐到同一 embedding 空间。
- **意图解析多层回退**：LLM 输出优先 → LLM 漏掉属性 → 类别从常见属性列表回退提取 → 属性从 query 中精确匹配提取。
- **Ollama**：LLM 和 VL 模型依赖 Ollama 本地推理。VL 默认 3 个 worker 并行做二分类验证，每个 worker 持有独立 `ChatOllama` 实例。粗排阶段控制候选数量避免过载。
- **SSE 流式进度**：Web 前端通过 Server-Sent Events 接收检索进度，VL 精排阶段实时推送当前验证的图像名和进度（`web_app/app.py` 中用 `asyncio.Queue` 桥接同步 pipeline 与异步 SSE）。
- **chat/chat_structured 共享 _chat_impl()**：核心路由逻辑集中在 `_chat_impl()` 私有方法中；`chat()` 提取 `output` 字段返回；`chat_structured()` 返回完整结构化数据 + 支持 progress_callback。CLI 和 Web 两个入口互不影响，修改路由只需改一处。
- **图像数据**：期望图像位于 `test_images/` 目录下。`data_load.py` 提供图像发现、计数和采样工具。
