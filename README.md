# 多模态图像检索系统 (Multi-Modal Image Retrieval System)

基于程序化路由编排的多模态图像检索系统，支持自然语言查询、LLM 意图识别、两阶段检索（粗排 CLIP + 精排 VL_Refine）。路由决策由代码确定性执行，LLM 专注于意图理解、视觉精排和回复生成。

## 核心特性

- **自然语言交互，无需关键词匹配**
- **智能意图识别**：LLM 自动提取类别、数量、检索方式、属性条件。模型通过 `INTENT_MODEL` 环境变量配置（默认 `qwen3:8b`，推荐 `qwen3:14b`）
- **两阶段检索**：粗排（CLIP 类别匹配）+ 精排（VL 模型逐一验证属性条件）
- **VL 精排**：VL 模型对候选图像验证属性（颜色、姿态、场景等），加权融合分数。模型通过 `VL_MODEL` 环境变量配置（默认 `qwen3-vl:8b`）
- **程序化路由**：路由决策由代码确定性执行（`if attributes:`），不依赖 LLM 自行选择，消除小模型路由不可靠的问题
- **中→英翻译层**：CLIP 为英文模型，查询编码前自动将中文关键词翻译为英文以提升检索准确度
- **共享离线管线**：常规检索与细粒度检索共用 CLIP 图像编码器构建的离线向量库
- **本地优先，云端可选**：默认全本地运行，模型可升级（见下方「架构演进」）

## 系统总体架构

```
用户输入 Query
    ↓
LLM 意图识别 (Ollama — INTENT_MODEL 可配置)
    ↓
├── 类别 + 数量 + 属性条件
    ↓
HasAttr?（代码决策，非 LLM 路由）
├── 否 → 常规检索
│        └── 中→英翻译 → CLIP文本编码 → 相似度计算 → 按数量Top-K
│
└── 是 → 细粒度两阶段检索
         ├── 粗排：中→英翻译 → CLIP文本编码 → VectorDB相似度 → TopK/卡阈值
         └── 精排：VL 模型逐一验证属性条件 → 硬过滤 → 加权融合 → Top-K

图像库 → CLIP图像编码器 → 离线向量库（共享）
         ↑ 中→英翻译层（CLIP为英文模型）
```

## 技术栈

- **LangChain**：LLM 调用和链式封装
- **Ollama**：本地 LLM 运行环境
- **Qwen3**：意图识别（可配置，默认 `qwen3:8b`）
- **Qwen2.5-VL / Qwen3-VL**：视觉语言模型，VL_Refine 精排（可配置）
- **CLIP (ViT-L/14)**：图像-文本匹配模型（粗排 + 离线编码）
- **PyTorch / Transformers**：深度学习推理框架

## 模块说明

### 1. 意图识别模块 (`intent_module/`)

使用 Qwen2.5-3B 模型分析用户查询，自动提取以下信息：
- **是否需要检索**：是/否
- **检索类别**：如"狗"、"熊"、"键盘"等（支持开放类别识别）
- **检索数量**：具体数字（如"2"）或模糊描述（如"很多"）
- **检索方式**：TopK（具体数字）或 卡阈值（模糊数量）
- **属性条件**：如"棕色"、"站立"、"清晰"、"全身"等颜色、姿态、场景描述

### 2. 常规检索模块 (`regular_retrieval_module/`)

适用于**无属性条件**的查询（如"狗的图片"、"2只猫"）：
- **CLIP 图像编码器**：基于 CLIP-ViT-Large，用于离线图像特征提取
- **离线阶段**：图像库 → CLIP 图像编码器 → 离线向量库
- **在线阶段**：查询 → CLIP 文本编码器 → 相似度计算 → 按数量 Top-K 返回

### 3. 细粒度检索模块 (`fine_grained_retrieval_module/`)

适用于**有属性条件**的查询（如"棕色的狗"、"站立的熊"），采用两阶段架构：

- **离线阶段**：图像库 → CLIP 图像编码器 → 离线向量库（与常规模块共享同一条管线）
- **在线粗排**：查询 → CLIP 文本编码器 → 相似度计算 → 检索策略（TopK / 卡阈值）
- **在线精排（VL_Refine）**：使用 Qwen2.5-VL 对粗排候选图像逐一验证属性条件，返回 0~1 的匹配分数，与粗排分数加权融合后重排序
- 新增**硬过滤机制**

### 4. 程序化路由编排

路由逻辑位于 `agent_pipeline/pipeline.py` 的 `chat()` 方法中，由代码确定性执行：

1. `IntentRecognitionModule.analyze_intent()` 解析用户查询
2. `need_retrieval=False` → 直接回复"不需要检索"
3. `attributes` 非空 → 细粒度两阶段检索（自动触发 VL_Refine）
4. `attributes` 为空 → 常规检索（单阶段 CLIP）

LLM 不参与路由决策，只负责意图识别、VL 精排和回复格式化三个核心环节。

## 架构演进：本地模型 → SOTA 多模态 API

当前系统默认使用本地小模型（qwen2.5:3b + qwen2.5vl:3b），核心瓶颈在 **VL 精排**：3B 参数的视觉语言模型对细粒度属性（"站立"、"棕色"、"室外"）的判断能力有限，经常给出全 0 的评分，导致细粒度检索路径几乎不可用。

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

**CLIP 仍然保留**做粗筛（本地、快速、零成本），**多模态 API 替代小模型**做精排和理解。这是“本地负责快，云端负责准”的混合策略。

### 对比

| 维度 | 当前（qwen2.5vl:3b 本地） | SOTA 多模态 API |
|------|--------------------------|----------------|
| VL 精排准确度 | 3B 小模型，属性判断不稳定 | 极高，能区分"站立"vs"坐着"、"棕色"vs"褐色" |
| 意图理解 | 不可靠，需要 parser 多层回退 + 代码修补 | 一次调用，结构化输出，无需 parser |
| 复杂查询 | 只能处理简单属性 | 能理解"草地上奔跑的金毛犬"等多条件组合 |
| 延迟（精排） | 每张 ~30s（本地推理） | 每张 ~1-2s（网络 + 推理） |
| 成本 | 零（本地 GPU） | 按 token 计费，每次查询约几分钱 |
| 隐私 | 图片不出本地 | 图片上传云端（精排阶段） |

### 切换方式

1. 环境变量改为 API 模型名
2. 将 `ChatOllama` 替换为对应的 API ChatModel（如 `ChatAnthropic`）
3. VL 精排阶段改为将 base64 图片发送至 API 进行评分
4. 粗排（CLIP）和离线索引保持不变

### 模型升级（本地方案）

在不引入外部 API 的情况下，也可以升级本地模型以获得可观改善：

```powershell
# 拉取模型（默认已使用 qwen3:8b + qwen3-vl:8b，以下为更高配置）
ollama pull qwen3:14b
ollama pull qwen3-vl:8b

# 切换
$env:INTENT_MODEL="qwen3:14b"
$env:VL_MODEL="qwen3-vl:8b"
```

Qwen3 系列在中文理解、指令遵循和视觉推理上相比 Qwen2.5-3B 有质的提升，但精排准确度仍远不及 SOTA API。

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
| `clip_ViT` | CLIP 图像编码（离线向量库构建） | `python scripts/download_models.py --model clip_ViT` |
| `qwen2.5:3b` | 意图识别（Ollama） | `ollama pull qwen2.5:3b` |
| `qwen2.5vl:3b` | VL_Refine 精排（Ollama） | `ollama pull qwen2.5vl:3b` |

### 下载模型

#### Ollama 

```bash
# 国内镜像源
set OLLAMA_REGISTRY_MIRROR=https://ollama.modelscope.cn

# Qwen2.5-3B 
ollama pull qwen2.5:3b

# Qwen2.5-VL
ollama pull qwen2.5vl:3b
```

#### clip_ViT & Chinese_RoBERTa

```bash
# 使用国内镜像源下载所有模型（推荐）
python scripts/download_models.py --output-dir ./models

# 下载单个模型
python scripts/download_models.py --model clip_ViT

# 使用官方源下载
python scripts/download_models.py --mirror official

# 验证已下载的模型
python scripts/download_models.py --verify-only
```

## 使用方法

### 交互式对话

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




