# 多类别检索改进方案 - 综合评估计划

> 评估日期: 2026-04-28
> 评估目标: 分析将单类别检索扩展为多类别检索的可行性、影响和实施路径

---

## 1. 问题评估 (Problem Assessment)

### 1.1 多类别检索解决的具体问题

当前系统将用户查询解析为**单一类别** (如 category="狗")，对于涉及多个物体的查询存在以下缺陷：

- **语义碰撞丢失**: 查询 "狗和猫" 编码为单一 CLIP embedding 时，向量在语义空间中落在"狗"和"猫"之间的模糊区域，与两类图像的相似度都偏低，导致真实"狗"或"猫"的评分被稀释。
- **数量分布不可控**: 查询 "找3张狗和2张猫" 若编码为 "狗和猫"，无法控制每类各返回多少张。
- **属性无法分属**: 查询 "棕色的狗和白色的猫" 无法区分哪些属性对应哪个类别。

具体场景示例:
| 用户查询 | 当前行为 | 期望行为 |
|---------|---------|---------|
| "找3张狗和2张猫" | 编码"狗和猫"->TopK=5 | 狗x3 + 猫x2，各自独立检索 |
| "棕色的狗和白色的猫" | 属性["棕色","白色"]->VL验证 | 狗+棕色，猫+白色，各自独立精排 |
| "几张狗和猫的图片" | 编码"狗和猫"->卡阈值 | 按比例或等分返回两类 |

### 1.2 多类别查询的常见性分析

**当前测试集统计**:
- test_queries/test_data.py: 25条查询，**0条**是多类别 (0%)
- experiment/intent_experiment/intent_test_queries.json: 27条查询，**0条**是多类别 (0%)

**实际场景预判**:
- 对比型查询: "狗和猫哪个更像..." (可能在检索场景出现)
- 组合型查询: "找狗和猫的图片"、"给我看熊和大象"
- 数量分配型: "找2张狗和3张猫"
- 属性分配型: "棕色的狗和白色的猫"

**预估频率**: 在真实用户查询中，多类别查询可能占 **5%-15%**。占比较低但一旦出现，当前系统的体验严重退化。

### 1.3 Embedding 语义碰撞问题

当将 "狗和猫" 编码为单一文本 embedding:
1. Taiyi-CLIP (Chinese RoBERTa) 将整个短语编码为一个高维向量
2. 该向量在语义空间中落在"狗" embedding 和"猫" embedding 之间
3. 与图像库中每张图片的相似度计算关系:
   - sim(encoding("狗和猫"), real_dog_image) < sim(encoding("狗"), real_dog_image)
   - sim(encoding("狗和猫"), real_cat_image) < sim(encoding("猫"), real_cat_image)
4. 这导致两类候选的排名都低于应有的水平，容易混入不相关结果

**实验验证方法**: 使用 experiment/retrieval_eval/run_retrieval_eval.py 的框架，对比:
- module.retrieve(query="狗", top_k=10) 的 F1-score
- module.retrieve(query="狗和猫", top_k=10) 的 F1-score
预期 "狗和猫" 在"狗"类上的 F1 明显下降。

---
## 2. 影响分析 (Impact Analysis per Component)

### 2.1 Intent Prompt (intent_module/prompt_template.py)

**现状 (第8-39行)**: Prompt 指令要求 LLM 输出"检索类别：[单个物体]"，明确限制为单个类别。

**需要的变更**:
- 新增多类别输出格式指令，例如:
```
检索类别列表：[{"类别":"狗","数量":3,"属性":"棕色"},{"类别":"猫","数量":2,"属性":"白色"}]
```

**回归风险**: **中**
- LLM 在单类别查询上仍能正确输出单元素列表，兼容性可保证
- 但 prompt 变长、格式变复杂后，LLM 的格式遵循准确率可能下降
- 需要大量测试验证 prompt 修改后单类别查询的准确率不降低

**边界情况**:
- 单类别无数量、混合具体/模糊数量

### 2.2 Intent Parser (intent_module/parser.py)

**现状 (第44-60行)**:
```python
category_match = re.search(r'检索类别[：:]\s*([^
]+)', text)
# 提取单个 category 字符串
# fallback: 遍历 common_objects 匹配第一个
```

**需要的变更**:
- 支持解析多类别格式（JSON 数组或 CSV 格式）
- 返回结构从 str 改为 List[Dict] 或保持兼容性的同时增加 categories 字段
- fallback 逻辑也需要支持多类别提取

**回归风险**: **高** — parser 是管道的第一道关卡
- 建议方案: 保持 category 字段存在 (取第一个类别)，新增 categories 字段 (列表)

**边界情况**:
- 查询 "找一些狗和猫" -> 数量为 vague，需要等分
- 查询 "狗" -> 只有单类别，保持兼容
- 查询 "找3只棕色的狗和2只白色的猫" -> 属性分属

### 2.3 Agent Pipeline (agent_pipeline/pipeline.py)

**现状 (第106-174行)**:
- category 作为标量字符串使用
- 路由逻辑: if not category -> fallback, if attributes -> fine_grained, else -> regular
- regular_retrieval.retrieve(query=category) 传入单字符串
- fine_grained.online_retrieval(category=category) 传入单字符串

**需要的变更**:
- 新增多类别路由分支: if len(categories) > 1: multi_category_retrieval()
- 每个类别独立执行检索，结果合并
- 合并策略: **按类别分组排序** (先展示狗的结果块，再展示猫的结果块)

**回归风险**: **高** — 路由逻辑涉及整个决策树，新增分支必须确保不干扰现有分支
- chat() 和 chat_structured() 两个方法都需要修改，逻辑需保持一致

**边界情况**:
- 多类别 + 无属性: 每类走常规检索
- 多类别 + 部分有属性: 部分走细粒度，部分走常规
- 某一类检索结果为0: 部分返回

### 2.4 Regular Retriever (regular_retrieval_module/retriever.py)

**影响**: **低** — 接口不变，只需重复调用 N 次（每类别一次）

### 2.5 Online Retriever (fine_grained_retrieval_module/online_retriever.py)

**影响**: **低** — 同样只需重复调用 N 次

### 2.6 VL Refiner (fine_grained_retrieval_module/vl_models.py)

**现状**: refine() 使用单 category 字符串构建 VL prompt

**需要的变更**:
- 多类别 VL 精排需要携带每个候选对应的类别标签
- 候选池在粗排后需要保持类别归属信息
- VL prompt 需要动态注入对应类别

**回归风险**: **中** — refine() 接口需兼容新旧两种结果格式

**边界情况**:
- 一张图片可能同时被检索为多类别 -> 去重问题
- VL 调用量 = 单类候选数 x 类别数，N 倍增长

### 2.7 前端 (web_app/static/index.html)

**影响**: **低** — 纯展示层变更，不影响检索逻辑
- 路由信息栏展示多类别列表: "类别: 狗(x3), 猫(x2)"
- 结果按类别分组展示或标注类别归属

### 2.8 Agent Tools (agent_pipeline/tools.py)

**影响**: **中** — 两个 Tool 函数都需要支持多类别参数传递
- 如果多类别仅在 pipeline.py 中实现而 tools.py 不更新，功能不一致

---

## 3. 设计决策 (Design Decisions)

### 3.1 决策一: 数据结构设计

**选项 A: 扩展 intent dict，新增字段 (推荐)**
```python
{
    "category": "狗",              # 保持兼容（取第一项）
    "categories": [                 # 新增列表
        {"category": "狗", "count": 3, "attributes": ["棕色"]},
        {"category": "猫", "count": 2, "attributes": ["白色"]}
    ],
    "is_multi_category": True       # 标记位
}
```

**选项 B: 完全替换 category 为列表** — 不推荐，破坏向后兼容

**推荐: 选项 A**。向后兼容，不影响现有单类别代码分支。

### 3.2 决策二: 属性分属支持

**问题**: "棕色的狗和白色的猫" — 属性是分属还是共有？

**推荐: 选项 A (分属属性)**。这是多类别检索的核心价值之一。

### 3.3 决策三: 模糊数量分布策略

**推荐: 选项 A (均分)**。简单、可预测。
- dog=ceil(total/2), cat=floor(total/2)
- 默认 total=5 时: 狗3张, 猫2张

### 3.4 决策四: 结果合并策略

**推荐: 选项 A (按类别分组排序)**。类别归属清晰。

### 3.5 决策五: 多类别与细粒度路径的关系

| 场景 | 建议行为 |
|------|---------|
| 多类别 + 无属性 | 每类走常规检索，合并 |
| 多类别 + 所有类都有属性 | 每类走细粒度 (粗排+VL精排)，合并 |
| 多类别 + 部分有属性 | 有属性的走细粒度，无属性的走常规 |
| 多类别 + 跨类别属性共享 | 暂不支持 (不在MVP范围内) |

---

## 4. 实施范围估算 (Implementation Scope Estimate)

### 4.1 变更文件清单

| 文件 | 变更类型 | 行数估计 | 复杂度 |
|------|---------|---------|--------|
| intent_module/prompt_template.py | 修改 | ~10行 | 低 |
| intent_module/parser.py | 修改 | ~30行 | 中 |
| agent_pipeline/pipeline.py | 修改 | ~60行 | 高 |
| agent_pipeline/tools.py | 修改 | ~20行 | 中 |
| fine_grained_retrieval_module/vl_models.py | 修改 | ~20行 | 中 |
| web_app/static/index.html | 修改 | ~15行 | 低 |
| test_queries/test_data.py | 新增 | ~10行 | 低 |
| experiment/intent_experiment/intent_test_queries.json | 新增 | ~20行 | 低 |

### 4.2 总计估算

- **文件变更数**: 8-9 个
- **新增代码行数**: ~120-150 行
- **修改代码行数**: ~50-60 行
- **整体复杂度**: **中等偏上 (Medium-High)**

### 4.3 可复用 vs 需重写

**可复用 (无变更)**:
- CLIPEncoder.encode_text() — 编码单文本，多类别只需多次调用
- Retriever.retrieve() / OnlineRetriever.retrieve() — 接口不变
- OfflineIndexer — 共享向量库不变
- data_load.py — 图像路径发现不变

**需修改**: parser.py, pipeline.py, vl_models.py, prompt_template.py

**需新增**: 多类别结果合并算法 (pipeline 内)

---

## 5. 风险评估 (Risk Assessment)

### 5.1 回归风险: 单类别查询行为

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Prompt 变更导致单类别解析下降 | 中 | 高 | 保留原始 prompt 作为备选 |
| Parser 多路径判断 bug | 中 | 高 | 增加单元测试 + 回归测试 |
| Pipeline 路由逻辑被干扰 | 低 | 极高 | 保持路由分支独立 |
| VL Refine 接口变更影响单类别 | 低 | 高 | 接口向后兼容设计 |

**回归测试建议**:
- 运行现有 25 条 test_queries/test_data.py 全部用例
- 运行 experiment/intent_experiment/run_intent_experiment.py (27 条)
- 运行 experiment/retrieval_eval/run_retrieval_eval.py
- 对比变更前后的准确率数字

### 5.2 LLM Prompt 变更风险

| 风险 | 说明 |
|------|------|
| 格式复杂度增加 | LLM 可能输出格式错误的 JSON |
| 推理时间增加 | 更长 prompt -> 更多 token -> 更慢响应 |
| 小模型能力不足 | qwen3:8b 可能在复杂格式上准确率下降 |

**建议**: 在 experiment/intent_experiment/ 框架下先测试

**备选方案**: 保留两套 prompt, 由 classifier 判断是否需要多类别解析

### 5.3 性能风险

| 场景 | 当前 | 多类别 (N=3) | 倍数 |
|------|------|-------------|------|
| CLIP 文本编码 | 1 次 | N 次 | xN |
| 相似度计算 | 1 次 x 全库 | N 次 x 全库 | xN |
| VL 精排调用 | M 张候选 | MxN 张候选 | xN |
| 总响应时间 | T (s) | ~NxT (s) | xN |

### 5.4 VL 精排影响

- 多类别 + 属性场景: VL 调用从 15 次跳升到 30 次
- 预计耗时: VL 单次 ~5-15s, 30 次可能需要 150-450s 串行
- 并发优化: 并行执行多类别的粗排和 VL 精排

---

## 6. 推荐方案 (Recommendation)

### 6.1 是否实施: **有条件实施**

多类别检索是合理的功能扩展，解决当前系统的一个明确缺陷。但优先级取决于产品目标:
- **如果是面向通用用户的检索产品**: 建议实施 (用户体验提升)
- **如果是面向特定领域/单物体检索**: 可暂缓

### 6.2 最小可行范围 (MVP)

第一阶段仅支持以下核心场景:
1. **简单多类别**: "找3张狗和2张猫" — 每类走常规检索，按数量分组合并
2. **模糊多类别**: "找几张狗和猫" — 均分默认数量
3. **单类别兼容**: 不改变现有单类别查询行为

**MVP 暂不支持**:
- 属性分属 ("棕色的狗和白色的猫")
- 多类别 + 细粒度路径
- VL 精排跨类别

### 6.3 推荐的实施顺序

```
Phase 1 (MVP): 数据结构 + Parser + Pipeline 常规检索分支
  |- intent 数据结构扩展 (保持向后兼容)
  |- Parser 多类别解析 (LLM 输出 + fallback)
  |- Pipeline 多类别路由 + 结果合并 (仅常规检索)
  |- 测试: 多类别 test cases + 单类别回归
  |- 预计: 1-2 天

Phase 2: 前端展示
  |- 路由信息栏展示多类别
  |- 结果按类别分组展示
  |- 预计: 0.5 天

Phase 3: 细粒度路径支持
  |- 多类别 + 属性分属
  |- VL 精排按类别分组验证
  |- 预计: 1-2 天

Phase 4: 全面测试 + 实验验证
  |- 多类别 intent 实验数据 + 运行实验
  |- 端到端检索评测
  |- 预计: 1 天
```

### 6.4 前置条件

1. **测试数据集**: 新增至少 10-15 条多类别查询及其 ground truth
   - 简单多类别 (无属性): 5 条
   - 多类别 + 部分属性: 3 条
   - 多类别 + 所有属性: 2 条
   - 模糊数量多类别: 2 条
   - 边缘案例 (部分类别无结果): 2 条
2. **回归基线**: 记录当前系统的准确率指标
   - intent 准确率: 当前 ~90%+
   - 常规检索 F1-score (@K=5)
3. **Prompt 预实验**: 在 intent_experiment 框架下先验证多类别 prompt 格式可行性

### 6.5 不做 (Don't do)

- 不支持无限类别 (限制最大 4-5 类)
- 不支持嵌套类别 — 超出 scope
- 不支持类别层级展开 — 增加复杂度

---

## 附录: 关键代码位置

| 组件 | 文件路径 | 关键行号 |
|------|---------|---------|
| Intent Prompt | intent_module/prompt_template.py | 第8-39行 |
| Intent Parser | intent_module/parser.py | 第44-60行 |
| Pipeline 路由 | agent_pipeline/pipeline.py | 第106-174行 |
| Regular Retriever | regular_retrieval_module/retriever.py | 第19-63行 |
| Online Retriever | fine_grained_retrieval_module/online_retriever.py | 第18-76行 |
| VL Refiner | fine_grained_retrieval_module/vl_models.py | 第65-110行 |
| 前端渲染 | web_app/static/index.html | 第485-511行 |
| 测试数据 | test_queries/test_data.py | 第10-55行 |
| Intent 实验 | experiment/intent_experiment/run_intent_experiment.py | 全部 |
| 检索评测 | experiment/retrieval_eval/run_retrieval_eval.py | 全部 |
| Agent Tools | agent_pipeline/tools.py | 第38-107行 |
