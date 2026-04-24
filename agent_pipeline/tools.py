"""
Agent工具定义
封装各个检索模块为LangChain Tool
"""

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field


def create_tools(intent_module, regular_retrieval, fine_grained):
    """
    创建Agent所需的工具列表
    
    Args:
        intent_module: 意图识别模块实例
        regular_retrieval: 常规检索模块实例
        fine_grained: 细粒度检索模块实例
        
    Returns:
        工具列表
    """
    def run_regular_retrieval(query: str) -> str:
        """为 Agent 提供干净调用的包装层，支持属性条件自动传递"""
        try:
            # 尝试从query中分离属性（用|分隔）
            parts = query.split("|")
            main_query = parts[0].strip()
            attributes = [a.strip() for a in parts[1:] if a.strip()] if len(parts) > 1 else []
            
            res = regular_retrieval.retrieve(main_query, method="TopK", top_k=5)
            
            # 如果有属性条件，进行精排序（已支持分数融合）
            if attributes and "results" in res:
                # 从query中提取类别（简单处理）
                category = main_query.split("的")[0] if "的" in main_query else main_query.split("包含")[0] if "包含" in main_query else main_query
                res["results"] = regular_retrieval.refine_by_attributes(
                    res.get("results", []), category, attributes
                )
                res["refinement_applied"] = True
                res["attributes_used"] = attributes
            
            return str(res)
        except Exception as e:
            return f"常规检索错误: {str(e)}"
    
    def run_fine_grained_retrieval(args: str) -> str:
        """为 Agent 提供干净调用的包装层，解析多参数并支持属性条件自动传递"""
        try:
            parts = [x.strip() for x in args.split(",")]
            if len(parts) >= 2:
                cat = parts[0]
                method = parts[1]
                count = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
                
                # 解析属性（用|分隔，支持"无"表示无属性）
                attributes = []
                if len(parts) >= 4:
                    attr_part = parts[3]
                    if attr_part and attr_part != "无":
                        if "|" in attr_part:
                            attributes = [a.strip() for a in attr_part.split("|") if a.strip()]
                        else:
                            attributes = [attr_part.strip()]
                
                res = fine_grained.online_retrieval(
                    category=cat, 
                    method=method, 
                    target_count=count,
                    attributes=attributes
                )
                
                # 如果有属性条件，自动执行两阶段精排序（已支持分数融合）
                if attributes and "results" in res:
                    res["results"] = fine_grained.refine_by_attributes(
                        res.get("results", []), cat, attributes
                    )
                    res["refinement_applied"] = True
                    res["attributes_used"] = attributes
                
                return str(res)
            return "参数解析错误，必须给出 '类别,检索方式'"
        except Exception as e:
            return f"发生了错误: {str(e)}"
    
    # 定义IntentRecognition的输入参数结构
    class IntentRecognitionInput(BaseModel):
        query: str = Field(description="用户的查询文本，例如：'帮我找两只狗的图片'")

    def _run_intent(query: str) -> str:
        """运行意图识别"""
        return str(intent_module.analyze_intent(query))

    tools = [
        StructuredTool(
            name="IntentRecognition",
            func=_run_intent,
            description="意图识别工具。在任何多模态检索操作前，必须先调用此工具来解析用户的原本问题，它将返回是否需要检索、检索的特定类别、检索数量、检索方式（TopK/卡阈值）以及属性条件（如颜色、姿态、场景等）。",
            args_schema=IntentRecognitionInput
        ),
        Tool(
            name="RegularImageRetrieval",
            func=run_regular_retrieval,
            description="常规泛化图像检索工具。在用户需求不涉及精确的数量约束时（即无数量词），使用此工具进行文本到图像的泛意境匹配。输入参数：中文query文本，可附加属性条件用|分隔。示例：'狗的图片' 或 '狗的图片|棕色|站立'。属性条件会自动从IntentRecognition结果中提取并传递。"
        ),
        Tool(
            name="FineGrainedRetrieval",
            func=run_fine_grained_retrieval,
            description="细粒度图像检索工具。当用户意图涉及具体类别的数量（如：两只狗，多个键盘）或者需要精确实体搜索时，使用此工具。输入参数格式：'类别,检索方式,数量,属性1|属性2'。示例：'狗,TopK,2,棕色|站立' 或 '熊,卡阈值,很多' 或 '狗,TopK,5'（无属性）。属性条件会自动从IntentRecognition结果中提取并传递。"
        )
    ]
    
    return tools