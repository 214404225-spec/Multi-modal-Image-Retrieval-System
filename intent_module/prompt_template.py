"""
意图识别模块的Prompt模板定义
"""

from langchain_core.prompts import PromptTemplate

# 意图识别的Prompt模板
INTENT_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""你是一个智能意图识别助手。请分析以下用户查询的意图：
查询："{query}"

请按如下格式输出你的判断：
是否需要检索：[是/否]
检索类别：[如果需要图像检索，请识别出查询中涉及的主要物体或场景类别，否则填无]
检索数量：[如果有数量词，提取具体数字如"2"，或模糊描述如"很多"，否则填无]
检索方式：[TopK 或 卡阈值，参考规则：具体数字为TopK，模糊词汇如"一些/很多"为卡阈值]
属性条件：[如果有额外条件如颜色、姿态、场景等，提取关键词用逗号分隔，否则填无]
"""
)
