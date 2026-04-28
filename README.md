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

## 架构迭代：v0 → v1

### 初版架构 (v0.md)

初版采用人工路由：用户查询经 Qwen2.5-3B 意图识别后，由人工/LLM选择检索路径。细粒度路径中，VL 模型（Qwen2.5-VL-3B）在离线建库阶段处理所有图像，产出通用增强表示后存入向量库，再与 CLIP 文本编码做相似度匹配。

初版实验结果：
- 意图识别（类别+数量）准确率：86%
- 常规检索 F1：80%（CaFo 微调后 85%）
- 细粒度检索 F1：88%（LoRA 后 93%）

### 新版核心升级（v1.md）

**1. 意图识别模型升级**：Qwen2.5-3B → Qwen3-8B。Qwen3 在中文理解、指令遵循和结构化输出方面显著更强，配合 parser 多层回退逻辑，意图识别准确率从 86% 提升至 **90.9%**（涵盖类别、数量、属性条件）。

**2. 路由决策程序化**：初版由人工/ LLM 自行选择检索路径（小模型路由不可靠，存在 ~5% 路径选择错误），新版改为 `if attributes:` 确定性代码路由——LLM 只负责提取结构化意图，路由决策完全由代码执行，消除了路由级错误。

**3. VL 精排模型升级**：Qwen2.5-VL-3B → Qwen3-VL-8B，属性二分类误判率大幅降低。

**4. VL 使用范式升级：离线盲增强 → 在线靶向验证**：

| 维度 | 初版（离线盲增强） | 新版（在线靶向验证） |
|------|-------------------|---------------------|
| VL 工作时机 | 离线建库 | 在线检索 |
| VL 是否感知 query | 否——编码时不知道用户会问什么 | 是——逐条针对 query 属性做判断 |
| VL 产出 | 通用图像增强表示 | 属性二分类验证结果（是/否） |
| 对属性的分辨力 | 间接——属性语义可能被主导类别淹没 | 直接——显式检查每个属性条件 |
| 典型失效模式 | "棕色站立狗"→VL 编码主要捕捉"狗"，"棕色""站立"在后续 CLIP 匹配中信号弱 | VL 偶有单属性误判，但属性组合整体判断准确 |

即使 VL 模型不变，仅将 VL 从离线盲增强切换到在线靶向验证，也能带来显著的细粒度精度提升。两项改善正交叠加：模型升级提供更强的视觉推理能力，范式升级确保这个能力被用在正确的环节（query-aware 验证而非离线盲编码）。

### 新版实验结果

| 指标 | 初版 | 新版 | 提升幅度 | 提升来源 |
|------|------|------|----------|----------|
| 意图识别准确率 | 86%（类别+方式） | **90.9%（类别+数量+属性条件）** | +4.9% | Qwen3-8B 更强的中文理解 + 多层回退 parser |
| 常规检索 F1 | 80% | **86%** | +6% | 意图识别更准 → 类别提取更准确 → 检索 query 质量提升 |
| 细粒度检索 F1 | 88% | **90%** | +2% | VL 模型升级 + VL 范式升级 + 意图识别提升 + 代码路由无损耗 |

**关键推论：**

- **意图识别是系统入口，入口精度提升对所有下游任务都有放大效应。** 意图错了后面全错，入口从 86% 提升到 90.9%，直接转化为常规和细粒度检索的 F1 增益。
- **VL 精排是细粒度路径的天花板。** 模型升级和范式变化正交叠加，共同推动细粒度 F1 提升。
- **代码路由消除了路由级错误。** 初版 LLM 路由存在 ~5% 的路径选择错误，新版代码路由将此错误率降为 0。

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
- **精排阶段** (`vl_models.py`)：VL 模型（Qwen3-VL）对粗排候选逐一做二分类验证（是/否），VL 通过则保留并按 CLIP 粗排分排序，VL 不通过则剔除。精排后 TopK 按用户请求数量精确裁剪

**VL 精排策略演进：从分级评分加权融合到二分类0-1检验**

早期设计中 `refine_by_attributes()` 和 `VLRefiner.refine()` 曾有三个参数——`alpha`（CLIP 权重，默认 0.4）、`beta`（VL 权重，默认 0.6）、`min_vl_score`（硬阈值，默认 0.2）。对应加权融合策略：

```
final_score = alpha × CLIP_coarse_score + beta × VL_attribute_score
if VL_attribute_score < min_vl_score → 硬过滤
```

VL 分数采用**分级评分**，语义锚定在"图像满足了多少个属性条件"上。例如用户输入 3 个属性（"棕色、站立、全身"）：

| VL 分数 | 含义 | 示例 |
|---------|------|------|
| 0.2–0.4 | 满足 ~1 个属性 | 狗是棕色的，但坐着且只拍半身 |
| 0.4–0.7 | 满足 ~2 个属性 | 狗是棕色且站立，但只拍半身 |
| 0.8–1.0 | 满足全部属性 | 棕色站立全身的狗 |

这种设计的核心价值在于：
- **分数可解释**：每档对应具体的属性满足数量，不再是抽象的"匹配程度"
- **软偏好 + 硬底线兼顾**：`min_vl_score`（0.2）确保至少满足 1 个属性才进入排序，加权融合则让满足更多属性的图在排序中获得更高最终分，而不必要求全部满足才能被看见
- **支持多属性查询的天然语义**：属性越多，分档越细粒度，加权融合后的排序越能反映"部分满足 > 完全不满足"的直觉

实际落地时，**本地部署的 qwen3-vl:8b（8B 参数）能力不足以支撑可靠的分级评分**：小模型缺乏校准的置信度输出能力，即使 prompt 明确要求按属性满足数量输出分级分数，也会倾向于输出极端值（几乎总是 1.0 或 0.0），无法稳定区分"满足 1 个属性"还是"满足 2 个属性"。这导致两个后果：

1. **VL 分数退化为二分类** — 模型输出的数值分数无区分度（几乎全是 1.0 / 0.0），继续对它加权融合只会污染 CLIP 粗排分
2. **alpha / beta 失去意义** — 在模型输出极端化的前提下，alpha / beta 的值完全起不到作用，调整alpha / beta 也失去了意义

根据上述情况，不如明确要求模型做二分类（是/否），把任务匹配到模型真正能可靠执行的水平上。最终简化为当前方案：**VL 只做二分类（是/否），不输出分数**。VL 通过 → 保留原始 CLIP 粗排分；VL 不通过 → 直接剔除。

> **理想情况下**，如果使用 SOTA 多模态 API（如 GPT-4o、Claude Opus 4.7）做精排，分级评分方案仍然优于二分类——大模型有足够的校准能力输出有区分度的分数，能够在"全部满足"和"完全不满足"之间给出有意义的中间档位，避免多属性查询中因一个属性不满足就完全剔除候选。这也是架构演进方向中"混合架构"的动机之一。

### 4. 共享组件 (`shared/`)

- `CLIPEncoder` (`shared/clip_encoder.py`)：Taiyi-CLIP 双模型编码器。文本编码使用 Chinese RoBERTa（`models/Chinese_RoBERTa/`），图像编码使用 CLIP ViT-L/14（`models/clip_ViT/`），两者经投影对齐到同一 embedding 空间。提供 `encode_text()`、`encode_images()` 和 `get_logit_scale()`

### 5. 程序化路由编排

路由逻辑位于 `agent_pipeline/pipeline.py` 的 `chat()` 方法中，由代码确定性执行：

1. `IntentRecognitionModule.analyze_intent()` 解析用户查询
2. `need_retrieval=False` → 直接回复"不需要检索"
3. `need_retrieval=True` 且 `attributes` 非空 → 细粒度两阶段检索（自动触发 VL_Refine）
4. `need_retrieval=True` 且 `attributes` 为空 → 常规检索（单阶段 CLIP）
5. 未提取到类别 → 回退：用原始查询做常规检索

**LLM 不参与路由决策，只负责意图识别、VL 精排和回复格式化三个核心环节**。

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

当前系统默认使用本地模型（qwen3:8b + qwen3-vl:8b），核心瓶颈在 **VL 精排**：8B 参数的视觉语言模型对细粒度属性（"站立"、"棕色"、"室外"）的判断能力有限，且限于性能无法使用分级评分，复杂属性组合下必然存在漏判。

### 理想架构：CLIP + SOTA 多模态 API 混合

如果接入 Claude Opus 4.7 / GPT-4o / Gemini 2.5 Pro 等云端多模态 API，架构变为：

```
用户查询 → 多模态 API（意图理解，自动结构化输出，无需 parser）
               ↓
        CLIP 粗排（本地）
               ↓
        Top-K 候选 → 多模态 API 精排
               ↓
        多模态 API 格式化回复
```

**CLIP 仍然保留**做粗筛（本地、快速、零成本），**多模态 API 替代小模型**做精排和理解。这是"本地负责快，云端负责准"的混合策略。

### 对比

| 维度 | 当前（qwen3-vl:8b 本地） | SOTA 多模态 API |
|------|--------------------------|----------------|
| VL 精排准确度 | 8B 小模型，复杂属性组合有漏判 | 极高，能区分"站立"vs"坐着"、"棕色"vs"褐色" |
| 意图理解 | 较可靠，parser 多层回退兜底 | 一次调用，结构化输出，无需 parser |
| 复杂查询 | 能处理中等复杂度属性，单属性效果最好 | 能理解"草地上奔跑的金毛犬"等多条件组合 |
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

### CLIP 编码器：**最值得微调**

工业界对 CLIP 架构做领域微调是**标准操作**。Taiyi-CLIP 的通用预训练数据与实际业务图像分布存在 domain gap：

- **电商**：商品图是白底棚拍，CLIP 预训练数据中大量是自然场景，零样本精度可能掉 10-20 个点
- **安防**：监控视角、低分辨率、红外图像，CLIP 预训练几乎未覆盖
- **医疗**：CT/X 光与自然图像的特征分布完全不同，不微调不可用

| 方法 | 成本 | 效果 | 适用场景 |
|------|:---:|:---:|------|
| LoRA 微调 CLIP 图像塔 | 低（4K 图即可） | +5-10% F1 | 领域图像分布偏移 |
| 对比学习续训 | 中（需 10K+ 图文对） | +8-15% F1 | 有大量领域标注 |
| 全量微调 | 高（需 50K+ 图文对） | +10-20% F1 | 极端 domain gap（医疗等） |

**CaFo 微调实践（v0 已采用）**：本系统在初版中已使用 CaFo（Cascade Foundation Models）对 Taiyi-CLIP 进行了微调，将常规检索 F1 从 80% 提升至 85%。CaFo 的核心操作是**冻结 Taiyi-CLIP 全部权重**（文本塔 Chinese RoBERTa + 图像塔 ViT-L/14 + 投影层均不更新），仅在 CLIP 输入端附加可训练的 prompt 向量（类似 CoOp），通过优化 prompt embedding 来适配下游领域数据，而非修改模型参数本身。

这种做法的优势：
- **保护预训练知识**：Taiyi-CLIP 在大规模中文图文数据上学到的对齐能力被完整保留，不会因领域数据量少而产生灾难性遗忘
- **参数效率极高**：可训练参数仅为一组 prompt 向量（通常几千个参数），远少于 LoRA 或全量微调
- **适合小样本场景**：4K 张领域图像即可获得显著提升，无需大量标注

训练完成后，推理时无需改动 CLIP 的任何推理代码——仅需在文本/图像编码前拼接上训练好的 prompt 向量即可。

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
| 用 API + 结构化输出 | 最主流——GPT-4o 的 structured output / function calling（不用写parser，function calling可以保证意图提取的格式正确性；Qwen3也支持function calling，但不一定比GPT-4o可靠） 已经足够可靠，不再需要自己微调 |

**本项目当前的 prompt 模板 + parser 多层回退本质是 prompt engineering 方案，工业界更倾向用大模型 API 的 native structured output 替代手写 parser**。

### 总结

```
值得微调的：  CLIP 编码器       ← 领域适应，ROI 最高
通常不微调的： VL 精排模型       ← 换更大模型比微调划算
看情况的：    意图识别 LLM       ← API 足够好就用 API，固定场景才微调小模型
```

## 已知限制
**VL 精排精度仍不及 SOTA API**：Qwen3vl:8b本地部署，性能有限。**解决方案时换用 SOTA API**。

**不支持专名/人名检索**：Taiyi-CLIP 文本编码器对专有名词（人名、地名、品牌等）无感知——这些 token 在训练语料中极少出现或不存在，无法编码为有意义的 embedding 向量。当用户查询包含专名时，结果会与原始专名无实际关联。同时 TopK 路径不设最低相似度阈值，导致低分匹配照常返回。**解决方案是引入额外的人脸标注、图像标签或关键词索引等信号源**。

**用户输入Query包含多个属性条件**：多属性输入VL模型，只输出二分类的0-1值，不会根据多个属性条件逐一判断有几个属性条件满足或哪个属性条件满足；单属性输入VL模型在本系统中表现最好。**解决方案是换用GPT-4o，将VL精排阶段的二分类评分改为分级评分**。

**用户输入Query包含多个类别**：当多个类别被编码为一个向量后，向量不单独代表某个类别而是代表多个类别的组合，检索结果会偏向多个类别共现的场景，而不是单独得分高的某个类别。**解决方案是意图识别模块新增拆分多类别独立检索后再合并**，但可能 Qwen3:8b 意图识别准确率会下降，涉及VL精排时检索时间可能会大幅增加。

## 评测数据集

以下数据集可用于评测本系统的意图识别、粗排召回和 VL 精排精度。按与本系统的匹配度（中文查询 + 多属性 + 图像检索）分为三层。

### 第一层：原生中文，可直接评测

| 数据集 | 规模 | 属性形式 | 适用评测环节 |
|--------|------|----------|-------------|
| **[MUGE](https://tianchi.aliyun.com/muge)** (阿里达摩院) | 250K 图文对，真实淘宝搜索日志 | 隐式：颜色/材质/风格嵌在 query 文本中（如"纯棉碎花吊带裙"） | 意图识别 + CLIP 粗排召回 |
| **[Product1M](https://github.com/zhanxlin/Product1M)** (ICCV 2021) | 118 万图文对，458 细粒度类目 | 品牌/产品名/功能等实例级属性 | 细粒度检索整体 F1 |
| **[FashionMT](https://github.com/PKU-ICST-MIPL/MAI_ICLR2025)** (北大, ICLR 2025) | 多轮交互，中英混合 | 显式：颜色/图案/袖长/领型等，支持属性回溯 | VL 多属性精排验证 |
| **[MERIT](https://merit-2025.github.io)** (NeurIPS 2025) | 320K query，含中文 | 多条件交织（多图+多文本属性） | 复杂多属性组合评测 |

### 第二层：学术标准 CIR 数据集（英文，可翻译适配）

| 数据集 | 规模 | 属性类型 | 特点 |
|--------|------|----------|------|
| **[FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)** (CVPR 2019) | 30K triplet（参考图 + 属性修改文本 → 目标图） | 显式属性修改：颜色/材质/款式/图案 | CIR 标准 benchmark，三类服装 |
| **[CIRR](https://github.com/Cuberick-Orion/CIRR)** (ICCV 2021) | 36K triplet，开放域 | 自然语言修改文本，隐式含属性 | 开放域，非限定服装 |
| **[CIRCO](https://github.com/miccunifi/CIRCO)** (ECCV 2024) | 1020 query，多 ground truth | 语义类别标注，自然语言修改 | 首个多 ground truth CIR 数据集 |
| **[COCO-Facet](https://arxiv.org/abs/2505.15877)** (NeurIPS 2025) | 9112 query，8 类属性 | 动物/物体/场景/姿态/时刻/人数/天气/材质 | 专门测试属性聚焦检索能力 |

### 第三层：属性标注数据集（可构建评测 ground truth）

这些数据集提供结构化的图像属性标签，可用来构造"属性条件查询 → 标注匹配图像"的评测对。

| 数据集 | 规模 | 属性空间 | 适用场景 |
|--------|------|----------|---------|
| **[PACO](https://github.com/facebookresearch/paco)** (Meta, CVPR 2023) | 641K part mask，75 类物体 | 55 属性（29 颜色 + 10 图案 + 13 材质 + 3 反射） | 评测 VL 对颜色/材质属性的判别准确率 |
| **[DeepFashion2](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)** (CUHK) | 800K+ 图像 | 50 类目 + 1000 描述属性 | 服装属性最全，可构造"红色长袖V领连衣裙"类 query |
| **[VAW](https://github.com/adobe-research/vaw_dataset)** (Adobe, CVPR 2021) | 260K object mask | 620 属性（颜色/形状/材质/状态/活动） | 属性种类最多，覆盖面广 |
| **[UPAR](https://chalearnlap.cvc.uab.cat/dataset/45/description/)** (WACV 2024) | 多域行人数据 | 40 二值属性（年龄/性别/发色/衣物颜色/配件） | 行人属性检索专用 |

### 分环节评测建议

- **意图识别** → 用 MUGE query 集评测 `attributes` 提取的准确率和召回率
- **粗排召回** → 用 DeepFashion2 属性标签构造"类别 + 属性条件"查询，评测粗排阶段召回上限
- **VL 精排精度** → 用 PACO 或 COCO-Facet，将属性标签转为二分类 prompt，统计 VL "是/否"判断的 precision/recall

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
| 粗排候选倍数 | `agent_pipeline/pipeline.py` | count × 3（保底 15） |
| 粗排阈值（卡阈值）| `fine_grained_retrieval_module/online_retriever.py` | 0.8 + 截断至 60 |
| 磁盘缓存 | `regular_retrieval_module/offline_indexer.py` | `cache/image_features.pt` |
| Top-K 默认值 | `pipeline.py:_resolve_top_k()` | 5 |

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


