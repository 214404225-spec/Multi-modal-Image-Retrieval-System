"""
意图识别模块主类
提供基于LLM的用户意图识别功能
"""

from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from .prompt_template import INTENT_RECOGNITION_PROMPT
from .parser import parse_output
from .constants import INTENT_MODEL_NAME


class IntentRecognitionModule:
    """
    意图识别模块：判断用户查询是否需要图像检索，并提取检索类别和检索方式。
    """
    def __init__(self, model_name: str = None, llm=None):
        if model_name is None:
            model_name = INTENT_MODEL_NAME
        self.llm = llm if llm is not None else ChatOllama(
            model=model_name,
            temperature=0.0,
            timeout=120
        )

        # 构建 Chain
        self.chain = INTENT_RECOGNITION_PROMPT | self.llm | StrOutputParser()

    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """对外部提供的接口，调用Chain进行分析并解析结果。"""
        try:
            # 执行大模型推理
            raw_response = self.chain.invoke({"query": query})
            # 解析输出结果
            result = parse_output(raw_response, query)
            return result
        except Exception as e:
            return {"error": f"意图识别失败: {str(e)}"}