# 多模态图像检索系统 (Multi-Modal Image Retrieval System)

基于 LangGraph 搭建的智能多模态图像检索系统，支持自然语言查询、意图识别、两阶段检索（粗检索+精排序）等功能。

## 系统架构

```
用户输入 Query
    ↓
Qwen2.5-3B 意图识别 (Ollama)
    ↓
├── 类别 + 数量 + 检索方式 + 属性条件
    ↓
智能路由
├── 有数量 → 细粒度检索模块
├── 无数量 → 常规检索模块
└── 不需要检索 → 直接回复
    ↓
两阶段检索（如有属性条件）
├── 第一阶段：粗检索（类别匹配）
└── 第二阶段：精排序（属性重排序）
    ↓
返回检索结果
```

## 核心特性

- **智能意图识别**：使用 Ollama 本地运行 Qwen2.5-3B，自动提取类别、数量、检索方式、属性条件
- **两阶段检索**：粗检索（类别匹配）+ 精排序（属性重排序）
- **零样本泛化**：利用 CLIP 的零样本能力检索未见类别
- **双路检索**：常规检索（泛化匹配）+ 细粒度检索（精确匹配）
- **本地部署**：所有模型均可本地运行，无需外部 API
- **模块化设计**：采用包结构组织代码，便于维护和扩展
- **离线模型加载**：支持预下载模型到本地，启动无需联网

## 文件结构

```
pipeline/
├── intent_module/                 # 意图识别模块包
│   ├── __init__.py
│   ├── constants.py               # 常量定义（中文数字映射）
│   ├── prompt_template.py         # Prompt模板
│   ├── parser.py                  # 输出解析器
│   └── module.py                  # 主模块类
│
├── regular_retrieval_module/      # 常规检索模块包
│   ├── __init__.py
│   ├── constants.py               # 常量定义
│   ├── text_encoder.py            # 中文RoBERTa文本编码器
│   ├── image_encoder.py           # clip_ViT图像编码器
│   ├── offline_indexer.py         # 离线索引器
│   ├── attribute_refiner.py       # 属性精排序器
│   ├── retriever.py               # 检索器
│   └── module.py                  # 主模块类
│
├── fine_grained_retrieval_module/ # 细粒度检索模块包
│   ├── __init__.py
│   ├── constants.py               # 常量定义
│   ├── vl_models.py               # VL模型管理器
│   ├── clip_encoder.py            # CLIP编码器
│   ├── offline_indexer.py         # 离线索引器
│   ├── attribute_refiner.py       # 属性精排序器
│   ├── online_retriever.py        # 在线检索器
│   └── module.py                  # 主模块类
│
├── agent_pipeline/                # Agent编排包
│   ├── __init__.py
│   ├── tools.py                   # 工具定义
│   ├── pipeline.py                # 主流程
│   └── main.py                    # 交互入口
│
├── test_20_queries/               # 测试包
│   ├── __init__.py
│   ├── test_data.py               # 测试数据
│   ├── runner.py                  # 测试运行器
│   └── main.py                    # 测试入口
│
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

## 环境要求

- Python 3.9+
- Ollama（用于运行 Qwen2.5-3B, Qwen2.5-VL-3B）
- CUDA GPU（可选，用于加速模型推理）

## 安装

### 1. Ollama 模型部署

```bash
# 国内镜像源备选：
set OLLAMA_REGISTRY_MIRROR=https://ollama.modelscope.cn

# 拉取 Qwen2.5-3B 模型
ollama pull qwen2.5:3b

# 拉取 Qwen2.5-VL 多模态模型（若没有拉取，会使用Transformers拉取）
ollama pull qwen2.5vl:3b
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 交互式对话

```bash
python -m agent_pipeline.main
```

输入自然语言查询，例如：
- "帮我找两只狗的图片"
- "找很多熊的照片"
- "搜索狗的图片"
- "找10张狗的全身图片"

### 运行测试

```bash
python -m test_20_queries.main
```

## 模块说明

### 1. 意图识别模块 (`intent_module/`)

使用 Qwen2.5-3B 模型分析用户查询，自动提取以下信息：
- **是否需要检索**：是/否
- **检索类别**：如"狗"、"熊"、"键盘"等（支持开放类别识别）
- **检索数量**：具体数字（如"2"）或模糊描述（如"很多"）
- **检索方式**：TopK（具体数字）或 卡阈值（模糊数量）
- **属性条件**：如"棕色"、"站立"、"清晰"、"全身"等

### 2. 常规检索模块 (`regular_retrieval_module/`)

适用于**无数量词**的泛化查询：
- **中文RoBERTa文本编码器**：基于 Taiyi-CLIP-Roberta，专为中文语义理解优化
- **clip_ViT图像编码器**：基于 CLIP-ViT-Large，用于图像特征提取
- **离线阶段**：图像库 → clip_ViT图像编码器 → 离线向量库
- **在线阶段**：查询 → 中文RoBERTa文本编码 → 相似度计算 → 检索
- **属性精排序**：支持根据属性条件对结果进行重排序

### 3. 细粒度检索模块 (`fine_grained_retrieval_module/`)

适用于**有数量词**的精确查询：
- **离线阶段**：图像库 → Qwen2.5-VL → 类别标签+数量 → CLIP文本编码 → 离线向量库
- **在线阶段**：查询 → CLIP文本编码 → 相似度计算 → 检索
- **核心功能**：
  - 零样本泛化：支持检索未见类别
  - 属性精排序：根据属性条件重排序结果
  - Ollama VL 支持：可通过 Ollama 调用 Qwen2.5-VL

### 4. Agent 主流程 (`agent_pipeline/`)

使用 LangGraph ReAct Agent 编排：
1. 调用意图识别工具分析查询
2. 根据意图结果自动路由到合适的检索模块
3. 如有属性条件，自动执行两阶段检索
4. 组织结果并回复用户

## 查询格式

### 常规查询
```
"帮我找狗的图片"
"搜索包含熊的图像"
```

### 带数量的查询
```
"帮我找两只狗的图片"
"找5个键盘的图片"
"找很多熊的照片"
```

### 带属性条件的查询
```
"帮我找10张狗的全身图片"
"找5只棕色的猫"
"给我看一些清晰的键盘照片"
```

## 路由逻辑

| 查询类型 | 示例 | 路由目标 |
|---------|------|---------|
| 具体数量 | "两只狗"、"3个键盘" | 细粒度检索 (TopK) |
| 模糊数量 | "很多狗"、"一些熊" | 细粒度检索 (卡阈值) |
| 无数量词 | "狗的图片" | 常规检索 |
| 不需要检索 | "你好"、"谢谢" | 直接回复 |

## 属性条件两阶段检索

### 整体流程

```
用户输入："帮我找10张狗的全身图片，要清晰的"
    ↓
意图识别提取：category="狗", count=10, attributes=["全身", "清晰"]
    ↓
┌─────────────────────────────────────────────────┐
│ 第一阶段：粗检索（类别匹配）                      │
│ - 使用类别"狗"构建查询文本                        │
│ - CLIP编码查询文本和图像库                        │
│ - 计算相似度，召回候选结果                        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ 第二阶段：精排序（属性重排序）                    │
│ - 使用"狗的全身、清晰"构建属性查询                │
│ - 重新计算与属性条件的相似度                      │
│ - 按融合分数重排序，返回TopK结果                  │
└─────────────────────────────────────────────────┘
```

### 第一阶段：粗检索（类别匹配）

**实现位置**：
- 细粒度模块：`fine_grained_retrieval_module/online_retriever.py` 的 `retrieve()` 方法
- 常规模块：`regular_retrieval_module/retriever.py` 的 `retrieve()` 方法

**核心逻辑**：
```python
# 1. 构建搜索查询文本（仅使用类别）
search_query = f"{target_count}个{category}"  # 如 "10个狗"
# 或
search_query = f"包含{category}的图片"  # 如 "包含狗的图片"

# 2. CLIP编码查询文本
query_feature = self.clip_encoder.encode_text(search_query)

# 3. 遍历离线向量库，计算余弦相似度
for path, data in self.offline_indexer.get_db().items():
    db_vector = data["vector"]
    similarity = (query_feature @ db_vector.t()).item()
    results.append({"url": path, "score": similarity, ...})

# 4. 按分数排序并根据检索方式筛选
results = sorted(results, key=lambda x: x["score"], reverse=True)
if method == "TopK":
    results = results[:top_k]
```

**特点**：
- 使用类别进行语义匹配，不依赖精确标签
- 支持零样本泛化（未见类别也能检索）
- 返回初步排序的候选结果列表

### 第二阶段：精排序（属性重排序）

**实现位置**：
- 细粒度模块：`fine_grained_retrieval_module/attribute_refiner.py`
- 常规模块：`regular_retrieval_module/attribute_refiner.py`

**核心逻辑**：
```python
def refine(self, results, category, attributes, top_k=None, alpha=0.4, beta=0.6):
    # 1. 构建属性查询文本：类别 + 属性组合
    attr_query = f"{category}的{'、'.join(attributes)}"
    # 例如："狗的全身、清晰"
    
    # 2. 使用CLIP编码属性查询
    query_feature = self.clip_encoder.encode_text(attr_query)
    
    # 3. 对粗检索结果重新计算相似度
    for r in results:
        img_feat = self.clip_encoder.encode_image(r["url"])
        attr_score = (query_feature @ img_feat.t()).item()
        r["attr_score"] = attr_score  # 存储属性分数
        
        # 4. 加权融合分数
        category_score = r.get("score", 0)
        r["final_score"] = alpha * category_score + beta * attr_score
    
    # 5. 按融合分数重排序
    refined = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)
    return refined[:top_k] if top_k else refined
```

**特点**：
- 在粗检索结果基础上进行二次排序
- 使用更具体的属性描述文本重新编码
- 采用加权融合：`final_score = 0.4 * category_score + 0.6 * attr_score`
- 既保留类别相关性，又突出属性条件的匹配度

### Agent自动传递属性条件

**工具定义**（`agent_pipeline/tools.py`）：
- 常规检索工具：输入格式 `'中文查询|属性1|属性2'`
- 细粒度检索工具：输入格式 `'类别,检索方式,数量,属性1|属性2'`

**System Prompt指导**（`agent_pipeline/pipeline.py`）：
- Agent会自动从IntentRecognition结果中提取属性条件
- 如果属性条件为"无"或空，则省略属性部分
- 如果属性条件有值，则自动附加到检索参数中

## 技术栈

- **LangGraph**：Agent 编排框架
- **LangChain**：工具封装和链式调用
- **Ollama**：本地 LLM 运行环境
- **Qwen2.5-3B**：意图识别大模型
- **Qwen2.5-VL**：视觉语言模型
- **CLIP**：图像-文本匹配模型
- **中文RoBERTa**：Taiyi-CLIP 的文本编码器，基于 RoBERTa 架构，专为中文优化
- **clip_ViT**：Taiyi-CLIP 的图像编码器，基于 CLIP ViT 架构

## 模型下载与本地部署

系统支持将预训练模型下载到本地，后续启动无需联网。所有模型均可通过 `scripts/download_models.py` 脚本下载。

### 支持的模型列表

| 模型名称 | 用途 | 下载命令 |
|---------|------|---------|
| `Chinese_RoBERTa` | 中文文本编码（Taiyi-CLIP文本编码器） | `python scripts/download_models.py --model Chinese_RoBERTa` |
| `clip_ViT` | 图像编码（Taiyi-CLIP图像编码器） | `python scripts/download_models.py --model clip_ViT` |
| `qwen2.5:3b` | 意图识别（Ollama） | `ollama pull qwen2.5:3b` |
| `qwen2.5vl:3b` | 视觉语言模型（Ollama） | `ollama pull qwen2.5vl:3b` |

### 下载模型

```bash
# 使用国内镜像源下载所有模型（推荐）
python scripts/download_models.py --output-dir ./models

# 下载单个模型
python scripts/download_models.py --model Chinese_RoBERTa
python scripts/download_models.py --model clip_ViT

# 使用官方源下载
python scripts/download_models.py --mirror official
```

### 验证本地模型

```bash
# 仅验证已下载的模型
python scripts/download_models.py --verify-only
```

### 配置本地模型路径

在 `regular_retrieval_module/constants.py` 中配置本地模型缓存路径：

```python
LOCAL_MODEL_CACHE = {
    'Chinese_RoBERTa': './models/Chinese_RoBERTa',  # 中文文本编码器
    'clip_ViT': './models/clip_ViT',                # 图像编码器
}
```

### 模型加载说明

- **Chinese_RoBERTa文本编码器**：模型来源为 `IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese`，这是一个纯文本的 RoBERTa 模型（非 CLIP 模型），专为中文语义理解优化。使用 `[CLS]` token 的输出作为句子特征表示。启动时会自动检测本地模型，若未找到则尝试从网络下载。
- **clip_ViT图像编码器**：基于 CLIP-ViT-Large，用于图像特征提取。
- 建议首次运行前下载模型到本地，避免网络问题导致加载失败。

## 注意事项

1. 首次运行前建议下载模型到本地，避免网络问题导致加载失败
2. 使用 CUDA GPU 可显著提升推理速度
3. 图像检索需要预先建立离线索引（`offline_indexing` 方法）
4. 测试脚本中的图像路径为示例，实际使用需替换为真实图像路径

## 许可证

MIT License