"""
Agent工具定义
封装各个检索模块为LangChain Tool
"""

import os

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field


def _trim_result_paths(res: dict) -> dict:
    """将检索结果中的完整路径截短为文件名，避免暴露冗长的本地路径。"""
    if "results" not in res or not isinstance(res["results"], list):
        return res
    trimmed_results = []
    for r in res["results"]:
        trimmed = dict(r)
        if "url" in trimmed:
            trimmed["url"] = os.path.basename(trimmed["url"])
        trimmed_results.append(trimmed)
    return {**res, "results": trimmed_results}


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
        """常规检索：CLIP文本编码 → 相似度计算 → Top-K（无属性精排）"""
        try:
            # 先用意图识别验证是否真的需要检索
            intent = intent_module.analyze_intent(query)
            if not intent.get("need_retrieval", True):
                return str({"message": "不需要检索", "results": []})

            import re
            k = 5
            m = re.search(r'(\d+)\s*[张只幅个份条]', query)
            if m:
                k = int(m.group(1))
            res = regular_retrieval.retrieve(query, method="TopK", top_k=k)
            return str(_trim_result_paths(res))
        except Exception as e:
            return f"常规检索错误: {str(e)}"

    def run_fine_grained_retrieval(args: str) -> str:
        """
        细粒度两阶段检索：粗排（CLIP）+ 精排（VL_Refine）。
        解析多参数并支持属性条件自动传递到 VL_Refine。
        """
        try:
            parts = [x.strip() for x in args.split(",")]
            if len(parts) < 2:
                return "参数解析错误，必须给出 '类别,检索方式'"

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

            # 无属性条件时，回退到常规检索
            if not attributes:
                import re
                k = count if isinstance(count, int) else 5
                query = f"{count}个{cat}" if count else cat
                res = regular_retrieval.retrieve(query, method="TopK", top_k=k)
                return str(_trim_result_paths(res))

            k = count if isinstance(count, int) and method == "TopK" else 5
            res = fine_grained.online_retrieval(
                category=cat,
                method=method,
                target_count=count,
                top_k=k,
                attributes=attributes
            )

            # 有属性条件时，执行 VL_Refine 精排（两阶段检索的第二阶段）
            if "results" in res:
                res["results"] = fine_grained.refine_by_attributes(
                    res.get("results", []), cat, attributes
                )
                res["refinement_applied"] = True
                res["refinement_method"] = "VL_Refine"
                res["attributes_used"] = attributes

            return str(_trim_result_paths(res))
        except Exception as e:
            return f"发生了错误: {str(e)}"

    # 定义IntentRecognition的输入参数结构
    class IntentRecognitionInput(BaseModel):
        query: str = Field(description="用户的查询文本，例如：'帮我找两只棕色的狗的图片'")

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
            description="常规图像检索工具。当用户需求不涉及属性条件（如颜色、姿态、场景等描述）时使用。输入参数：直接传入用户原始查询文本。示例：'狗的图片'、'2只猫'。"
        ),
        Tool(
            name="FineGrainedRetrieval",
            func=run_fine_grained_retrieval,
            description="细粒度两阶段图像检索工具。当用户查询包含属性条件（如颜色、姿态、场景等）时使用。采用粗排（CLIP）+ 精排（VL视觉语言模型验证属性）的双阶段架构。输入参数格式：'类别,检索方式,数量,属性1|属性2'。示例：'狗,TopK,2,棕色|站立' 或 '熊,卡阈值,很多,褐色'。"
        )
    ]

    return tools
