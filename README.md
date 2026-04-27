# 多模态图像检索系统 (Multi-Modal Image Retrieval System)

基于程序化路由编排的多模态图像检索系统，支持自然语言查询、LLM 意图识别、两阶段检索（粗排 CLIP + 精排 VL_Refine）。路由决策由代码确定性执行，LLM 专注于意图理解、视觉精排和回复生成。

## 核心特性

- **自然语言交互**：中文查询直接编码，无需翻译（Taiyi-CLIP Chinese RoBERTa 原生支持）
- **智能意图识别**：LLM 自动提取类别、数量、检索方式、属性条件。模型通过 `INTENT_MODEL` 环境变量配置（默认 `qwen3:8b`）
- **两阶段检索**：粗排（CLIP 类别匹配）+ 精排（VL 模型二分类验证属性条件，硬过滤）
- **VL 精排**：VL 模型对候选图像逐一验证属性（是/否），通过则保留 CLIP 粗排分排序，不通过则剔除。模型通过 `VL_MODEL` 环境变量配置（默认 `qwen3-vl:8b`）
- **程序化路由**：路由决策由代码确定性执行（`if attributes:`），不依赖 LLM 自行选择，消除小模型路由不可靠的问题
- **共享离线管线**：常规检索与细粒度检索共用 CLIP 图像编码器构建的离线向量库
- **Web 前端**：FastAPI + SSE 流式推送，直观展示检索图像缩略图，支持灯箱查看大图和分数详情。浏览器打开即可使用，无需额外配置
- **本地优先**：默认全本地运行，模型可升级（见下方「架构演进」）

## 系统总体架构

```
浏览器 / CLI
    ↓
┌─ Web 前端 (FastAPI + SSE) ─┐  或  CLI (agent_pipeline.main)
│  web_app/app.py             │
│  static/index.html          │
└─────────────────────────────┘
    ↓
用户输入 Query
    ↓
LLM 意图识别 (Ollama — INTENT_MODEL 可配置)
    ↓
├── 类别 + 数量 + 属性条件
    ↓
HasAttr?（代码决策，非 LLM 路由）
├── 否 → 常规检索
│        └── Chinese RoBERTa 文本编码 → 相似度计算 → 按数量 Top-K
│
└── 是 → 细粒度两阶段检索
         ├── 粗排：Chinese RoBERTa 文本编码（仅类别）→ VectorDB 相似度 → TopK/卡阈值
         └── 精排：VL 模型逐一验证属性条件（是/否）→ 硬过滤 → 保留 CLIP 粗排分排序 → Top-K

图像库 → CLIP ViT 图像编码 → 离线向量库（共享）
         ↑ Taiyi-CLIP 双模型架构（文本：Chinese RoBERTa，图像：CLIP ViT-L/14）
```

## 技术栈

- **LangChain**：LLM 调用和链式封装
- **Ollama**：本地 LLM 运行环境
- **Qwen3**：意图识别（可配置，默认 `qwen3:8b`）
- **Qwen3-VL**：视觉语言模型，VL_Refine 精排（可配置，默认 `qwen3-vl:8b`）
- **Taiyi-CLIP (ViT-L/14 + Chinese RoBERTa)**：中文原生双模型编码——文本用 Chinese RoBERTa，图像用 CLIP ViT-L/14，经投影对齐到同一 embedding 空间
- **PyTorch / Transformers**：深度学习推理框架

## 模块说明

### 1. 意图识别模块 (`intent_module/`)

使用 Qwen3-8B 模型分析用户查询，自动提取以下信息：
- **是否需要检索**：是/否
- **检索类别**：如"狗"、"熊"、"键盘"等（支持开放类别识别）
- **检索数量**：具体数字（如"2"）或模糊描述（如"很多"）
- **检索方式**：对应检索数量，TopK（具体数字）或 卡阈值（模糊数量）
- **属性条件**：如"棕色"、"站立"、"清晰"、"全身"等颜色、姿态、场景描述

`parser.py` 包含多层回退逻辑：LLM 输出优先 → LLM 漏掉属性时从 query 精确匹配提取 → LLM 误判数量时从原始 query 提取数字覆盖。

### 2. 常规检索模块 (`regular_retrieval_module/`)

适用于**无属性条件**的查询（如"狗的图片"、"2只猫"）：
- **共享 CLIP 编码器** (`shared/clip_encoder.py`)：Taiyi-CLIP 双模型架构，Chinese RoBERTa 编码文本，CLIP ViT-L/14 编码图像
- **离线阶段** (`offline_indexer.py`)：图像库 → CLIP ViT 图像编码 → 离线向量库，支持磁盘缓存（`cache/image_features.pt`）
- **在线阶段** (`retriever.py`)：中文查询 → Chinese RoBERTa 文本编码 → 相似度计算 → 按数量 Top-K 返回

### 3. 细粒度检索模块 (`fine_grained_retrieval_module/`)

适用于**有属性条件**的查询（如"棕色的狗"、"站立的熊"），采用两阶段架构：

- **离线阶段**：共享常规检索模块的 CLIP 图像编码向量库
- **粗排阶段** (`online_retriever.py`)：仅用类别构建查询（属性留给 VL 精排），经 Chinese RoBERTa 编码，在共享向量库中做相似度搜索。粗排取 `count × 3`（保底 15，非整数 count 时保底 30）个候选。「卡阈值」方法先按 0.8 阈值过滤再截断至前 60
- **精排阶段** (`vl_models.py`)：VL 模型（Qwen3-VL）对粗排候选逐一做二分类验证（是/否），VL 通过则保留并按 CLIP 粗排分排序，VL 不通过则剔除（硬过滤阈值 0.2）。精排后 TopK 按用户请求数量精确裁剪

### 4. 共享组件 (`shared/`)

- `CLIPEncoder` (`shared/clip_encoder.py`)：Taiyi-CLIP 双模型编码器。文本编码使用 Chinese RoBERTa（`models/Chinese_RoBERTa/`），图像编码使用 CLIP ViT-L/14（`models/clip_ViT/`），两者经投影对齐到同一 embedding 空间。提供 `encode_text()`、`encode_images()` 和 `get_logit_scale()`

### 5. 程序化路由编排

路由逻辑位于 `agent_pipeline/pipeline.py` 的 `chat()` 方法中，由代码确定性执行：

1. `IntentRecognitionModule.analyze_intent()` 解析用户查询
2. `need_retrieval=False` → 直接回复"不需要检索"
3. `need_retrieval=True` 且 `attributes` 非空 → 细粒度两阶段检索（自动触发 VL_Refine）
4. `need_retrieval=True` 且 `attributes` 为空 → 常规检索（单阶段 CLIP）
5. 未提取到类别 → 回退：用原始查询做常规检索

LLM 不参与路由决策，只负责意图识别、VL 精排和回复格式化三个核心环节。

### 6. Web 前端模块 (`web_app/`)

基于 FastAPI 的 Web 服务，提供浏览器端的图像检索界面：

- **`app.py`**：FastAPI 应用，启动时创建 `MultiModalAgentPipeline` 单例。核心端点：
  - `POST /api/chat/stream`：SSE 流式检索，推送进度事件（`intent` → `retrieval` → `vl_refine` → `formatting`）→ 最终返回结构化 JSON（含图片路径、各阶段分数、路由信息）
  - `GET /api/health`：服务健康检查（pipeline 状态、模型信息、图像总数）
  - `/images/{filename}`：从 `test_images/` 静态提供图像文件
  - `/`：单文件前端页面
- **`static/index.html`**：单文件前端，零外部依赖。深色/浅色主题自动切换，响应式布局。搜索框 → 进度条（VL 精排阶段实时显示当前候选图像名）→ 缩略图结果网格（含分数角标）→ 灯箱大图查看详情
- **SSE 流式设计**：`asyncio.Queue` 桥接同步 pipeline（后台线程）与异步 SSE 流，VL 精排的逐候选进度实时推送到前端

Pipeline 新增 `chat_structured()` 方法（`chat()` 保持不变），支持 `progress_callback` 回调参数，将检索各阶段进度和结构化结果传递到 Web 层。

## 架构演进：本地模型 → SOTA 多模态 API

当前系统默认使用本地模型（qwen3:8b + qwen3-vl:8b），核心瓶颈在 **VL 精排**：8B 参数的视觉语言模型对细粒度属性（"站立"、"棕色"、"室外"）的判断能力有限，复杂属性组合下仍存在漏判。

### 理想架构：CLIP + SOTA 多模态 API 混合

如果接入 Claude Opus 4.7 / GPT-4o / Gemini 2.5 Pro 等云端多模态 API，架构变为：

```
用户查询 → 多模态 API（意图理解，结构化输出，无需 parser）
               ↓
        CLIP 粗排（本地，毫秒级扫 2000 张）
               ↓
        Top-K 候选（~10 张）→ 多模态 API 精排
               ↓
        多模态 API 格式化回复
```

**CLIP 仍然保留**做粗筛（本地、快速、零成本），**多模态 API 替代小模型**做精排和理解。这是"本地负责快，云端负责准"的混合策略。

### 对比

| 维度 | 当前（qwen3-vl:8b 本地） | SOTA 多模态 API |
|------|--------------------------|----------------|
| VL 精排准确度 | 8B 小模型，复杂属性组合有漏判 | 极高，能区分"站立"vs"坐着"、"棕色"vs"褐色" |
| 意图理解 | 较可靠，parser 多层回退兜底 | 一次调用，结构化输出，无需 parser |
| 复杂查询 | 能处理中等复杂度属性 | 能理解"草地上奔跑的金毛犬"等多条件组合 |
| 延迟（精排） | 每张 ~1-5s（本地推理） | 每张 ~0.1-1s（网络 + 推理） |
| 成本 | 零（本地 GPU） | 按 token 计费 |
| 隐私 | 图片不出本地 | 图片上传云端（精排阶段） |

### 切换方式

1. 环境变量改为 API 模型名
2. 将 `ChatOllama` 替换为对应的 API ChatModel（如 `ChatAnthropic`）
3. VL 精排阶段改为将 base64 图片发送至 API 进行评分
4. 粗排（CLIP）和离线索引保持不变

## 工业落地的微调策略

将系统拆分为三个可独立微调的组件，工业界对各组件的微调投入差异显著：

### CLIP 编码器：最值得微调

工业界对 CLIP 做领域微调是**标准操作**。CLIP 的通用预训练数据（LAION-400M）与实际业务图像分布存在 domain gap：

- **电商**：商品图是白底棚拍，CLIP 预训练数据中大量是自然场景，零样本精度可能掉 10-20 个点
- **安防**：监控视角、低分辨率、红外图像，CLIP 预训练几乎未覆盖
- **医疗**：CT/X 光与自然图像的特征分布完全不同，不微调不可用

| 方法 | 成本 | 效果 | 适用场景 |
|------|:---:|:---:|------|
| LoRA 微调 CLIP 图像塔 | 低（4K 图即可） | +5-10% F1 | 领域图像分布偏移 |
| 对比学习续训 | 中（需 10K+ 图文对） | +8-15% F1 | 有大量领域标注 |
| 全量微调 | 高（需 50K+ 图文对） | +10-20% F1 | 极端 domain gap（医疗等） |

### VL 精排模型：通常不微调

工业界对 VL 精排阶段的模型一般**不做微调**：

1. VL 做的是二分类验证（是/否），不是开放域理解——这个任务对预训练 VL 已经足够简单，微调收益有限
2. VL 模型参数量大（3B-8B），微调需要大量图像+属性标注数据，ROI 低
3. 与其微调 VL，不如把算力投到 CLIP 粗排上——粗排召回率提升 5%，对最终效果的影响远大于 VL 精度提升 5%

工业界的真实做法是：**VL 精排精度不够 → 换更强的 VL 模型**（如本地 Qwen3-VL → 云端 GPT-4o），而不是微调。这也是为什么上方「架构演进」中将 "CLIP 本地粗排 + SOTA API 精排" 列为理想架构。

### 意图识别 LLM：看场景

| 做法 | 场景 |
|------|------|
| 不微调，用大模型 | 查询多样性高、长尾表达多，直接用 GPT-4o / Qwen3-14B 的零样本能力 |
| 微调小模型 | 查询模式固定（如电商搜索总是"颜色+品类+款式"），微调 1-3B 模型做专用意图识别，成本低、延迟低 |
| 用 API + 结构化输出 | 最主流——GPT-4o 的 structured output / function calling 已经足够可靠，不再需要自己微调 |

本项目当前的 prompt 模板 + parser 多层回退本质是 prompt engineering 方案，工业界更倾向用大模型 API 的 native structured output 替代手写 parser。

### 总结

```
值得微调的：  CLIP 编码器       ← 领域适应，ROI 最高
通常不微调的： VL 精排模型       ← 换更大模型比微调划算
看情况的：    意图识别 LLM       ← API 足够好就用 API，固定场景才微调小模型
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 模型本地部署

为了避免网络问题导致的报错，建议预先下载所有要用到的模型。

### 支持的模型列表

| 模型名称 | 用途 | 下载命令 |
|---------|------|---------|
| `clip_ViT` | CLIP ViT 图像编码（离线向量库构建） | `python scripts/download_models.py --model clip_ViT` |
| `Chinese_RoBERTa` | Taiyi-CLIP 中文文本编码 | `python scripts/download_models.py --model Chinese_RoBERTa` |
| `qwen3:8b` | 意图识别（Ollama） | `ollama pull qwen3:8b` |
| `qwen3-vl:8b` | VL_Refine 精排（Ollama） | `ollama pull qwen3-vl:8b` |

### 下载模型

#### Ollama

```bash
# 国内镜像源
set OLLAMA_REGISTRY_MIRROR=https://ollama.modelscope.cn

# Qwen3-8B（意图识别）
ollama pull qwen3:8b

# Qwen3-VL-8B（VL 精排）
ollama pull qwen3-vl:8b
```

#### clip_ViT & Chinese_RoBERTa

```bash
# 使用国内镜像源下载所有模型（推荐）
python scripts/download_models.py --output-dir ./models

# 下载单个模型
python scripts/download_models.py --model clip_ViT
python scripts/download_models.py --model Chinese_RoBERTa

# 使用官方源下载
python scripts/download_models.py --mirror official

# 验证已下载的模型
python scripts/download_models.py --verify-only
```

## 使用方法

### Web 前端（推荐）

```bash
# 启动 Web 服务
python -m uvicorn web_app.app:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000`，输入中文自然语言查询即可：
- 搜索框输入查询，回车或点击「检索」按钮
- 检索进度实时显示（意图分析 → CLIP 粗排 → VL 精排进度条）
- 结果以缩略图网格展示，每张卡片标注相似度分数
- 点击任意图片打开灯箱，查看大图和详细分数（CLIP 粗排分、VL 精排分、最终分）
- 支持深色/浅色主题自动切换，响应式布局适配移动端

**SSE 流式设计**：服务端通过 Server-Sent Events 推送检索进度（`POST /api/chat/stream`），VL 精排阶段逐候选图像报告进度，前端实时渲染进度条和当前验证的图像名。

### 交互式命令行

```bash
python -m agent_pipeline.main
```

输入自然语言查询，例如：
- "帮我找棕色的狗的图片"（有属性条件 → 细粒度两阶段检索 + VL_Refine）
- "找站立的熊的照片"（有属性条件 → 细粒度两阶段检索 + VL_Refine）
- "搜索狗的图片"（无属性条件 → 常规检索）
- "找2只猫的图片"（无属性条件 → 常规检索，按数量 Top-K）

### 运行批量测试示例

```bash
python -m test_queries.main
```

## 已知限制

**VL 精排精度仍不及 SOTA API**：复杂属性组合（"棕色站立的狗"）的细粒度判断仍存在漏判。理想混合架构（CLIP 本地粗筛 + 云端多模态 API 精排）详见上方「架构演进」章节。

**不支持专名/人名检索**：CLIP 文本编码器对专有名词（人名、地名、品牌等）无感知——这些 token 在训练语料中极少出现或不存在，无法编码为有意义的 embedding 向量。当用户查询包含专名时，意图模块会将其退化为泛化类别（"庄方宜"→"人"），检索本质是在用泛化类别做相似度搜索。由于 Taiyi-CLIP 的零样本泛化能力，即便图像集中没有对应类别，模型也会在 embedding 空间中找出距离最近的图像（如将"人"映射到视觉最接近的灵长类动物），但结果与原始专名无实际关联。同时 TopK 路径不设最低相似度阈值，导致低分匹配照常返回。格式化 LLM 最后会将用户 query 中的专名与这些无关结果强行缝合，产生虚假的相关性陈述。如需支持专名检索，需引入额外的人脸标注、图像标签或关键词索引等信号源。

## 配置项

| 配置项 | 环境变量 / 位置 | 默认值 |
|------|------|---------|
| **意图识别 LLM** | `INTENT_MODEL` | `qwen3:8b` |
| **VL 模型** | `VL_MODEL` | `qwen3-vl:8b` |
| CLIP ViT 模型路径 | `*/constants.py`（基于 PROJECT_ROOT 计算） | `models/clip_ViT` |
| Chinese RoBERTa 路径 | `shared/clip_encoder.py` | `models/Chinese_RoBERTa` |
| 索引规模 | `REGULAR_INDEX_SIZE` | `-1`（全部图像） |
| HuggingFace 镜像 | `pipeline.py` / `HF_ENDPOINT` | `https://huggingface.co` |
| VL 精排策略 | `fine_grained_retrieval_module/vl_models.py` | 二分类（是/否），通过则保留 CLIP 粗排分 |
| VL 硬过滤阈值 | `vl_models.py` | 0.2 |
| 粗排候选倍数 | `agent_pipeline/pipeline.py` | count × 3（保底 15） |
| 粗排阈值（卡阈值）| `fine_grained_retrieval_module/online_retriever.py` | 0.8 + 截断至 60 |
| 磁盘缓存 | `regular_retrieval_module/offline_indexer.py` | `cache/image_features.pt` |
| Top-K 默认值 | `pipeline.py:_resolve_top_k()` | 5 |
