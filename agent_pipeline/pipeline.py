"""
Agent Pipeline主类
LangGraph ReAct Agent编排的主流程
"""

import os
import glob
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# 导入我们的自定义模块
from intent_module import IntentRecognitionModule
from regular_retrieval_module import RegularRetrievalModule
from fine_grained_retrieval_module import FineGrainedRetrievalModule

from .tools import create_tools

# 图像库路径配置
# 获取项目根目录（pipeline目录的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")

def get_all_image_paths(directory: str) -> list:
    """获取目录下所有图像文件的绝对路径"""
    if not os.path.exists(directory):
        print(f"[警告] 图像目录不存在: {directory}")
        return []
    
    # 支持的图像格式（使用集合去重，避免Windows下大小写不敏感导致的重复匹配）
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = set()
    for ext in extensions:
        paths.update(glob.glob(os.path.join(directory, ext)))
    paths = list(paths)
    
    # 转换为绝对路径并排序
    paths = [os.path.abspath(p) for p in paths]
    paths.sort()
    return paths


class MultiModalAgentPipeline:
    """
    全流程多模态 Agent 工作流。使用 LangChain 的 Tool 封装三种不同的模块。
    通过 Ollama 本地运行 qwen2.5:3b 模型自主编排调用合适的工具完成复杂的用户查询检索任务。
    
    路由逻辑（基于数量明确性路由）：
    - 有具体数量（如"两只狗"、"3个键盘"）→ 细粒度检索
    - 有模糊数量（如"很多狗"、"一些熊"）→ 细粒度检索
    - 无数量词，仅有类别（如"狗的图片"）→ 常规检索
    - 不需要检索 → 直接回复
    """
    def __init__(self, model_name: str = "qwen2.5:3b"):
        # --- 1. 基础建设：初始化各个检索器模块 ---
        print(f"[INFO] 使用 Ollama 模型: {model_name}")
        self.intent_module = IntentRecognitionModule(model_name=model_name)
        self.regular_retrieval = RegularRetrievalModule()
        self.fine_grained = FineGrainedRetrievalModule()
        
        # 获取真实本地图像库路径
        all_image_paths = get_all_image_paths(TEST_IMAGES_DIR)
        print(f"[INFO] 从 {TEST_IMAGES_DIR} 加载了 {len(all_image_paths)} 张图像")
        
        # 离线索引配置：可通过环境变量控制索引数量
        # REGULAR_INDEX_SIZE: 常规检索模块索引数量（默认全部）
        # FINE_GRAINED_INDEX_SIZE: 细粒度检索模块索引数量（默认20，VL模型处理较慢）
        regular_index_size = int(os.environ.get("REGULAR_INDEX_SIZE", len(all_image_paths)))
        fine_grained_index_size = int(os.environ.get("FINE_GRAINED_INDEX_SIZE", "20"))
        
        if all_image_paths:
            # 常规检索模块：索引全部图像（或通过环境变量控制）
            regular_sample_size = min(regular_index_size, len(all_image_paths))
            regular_sample_paths = all_image_paths[:regular_sample_size]
            print(f"[INFO] 开始常规检索模块离线索引（共{len(regular_sample_paths)}张，总计{len(all_image_paths)}张可用）...")
            self.regular_retrieval.offline_indexing(regular_sample_paths)
            
            # 细粒度检索模块：使用真实本地图像进行离线索引
            # 注意：细粒度检索需要VL模型逐张识别图像，4000张图像可能需要很长时间
            # 默认只索引前20张，可通过环境变量 FINE_GRAINED_INDEX_SIZE 调整
            fine_grained_sample_size = min(fine_grained_index_size, len(all_image_paths))
            sample_paths = all_image_paths[:fine_grained_sample_size]
            print(f"[INFO] 开始细粒度检索模块离线索引（共{len(sample_paths)}张，总计{len(all_image_paths)}张可用）...")
            self.fine_grained.offline_indexing(sample_paths)
        else:
            print(f"[警告] 未找到图像，检索功能将不可用")

        # --- 2. 封装 LangChain Tool ---
        self.tools = create_tools(self.intent_module, self.regular_retrieval, self.fine_grained)

        # --- 3. 配置核心大脑：Agent LLM (使用 Ollama) ---
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0
        )
        
        # --- 4. 配置 Agent 编排框架 (LangGraph ReAct) ---
        system = '''你是一个高效权威的"多模态视觉检索调度主控"。
        你的任务是协助用户找到想要的图像。
        请务必遵循以下步骤逻辑：
        1. 总是先使用 `IntentRecognition` 判断意图。
        2. 根据意图识别结果自动路由：
        - 如果意图结果中"需要检索"为"是"，且"数量"字段有值（具体数字或模糊描述），调用 `FineGrainedRetrieval` 工具。
          参数格式：'类别,检索方式,数量,属性1|属性2'
          其中类别、检索方式、数量来自IntentRecognition结果。
          如果"属性条件"为"无"或空，则省略属性部分，仅传'类别,检索方式,数量'。
          如果"属性条件"有值（如["棕色", "站立"]），则传'类别,检索方式,数量,棕色|站立'。
        - 如果意图结果中"需要检索"为"是"，但"数量"字段为"无"或None，调用 `RegularImageRetrieval` 工具。
          参数格式：'中文查询文本|属性1|属性2'
          其中查询文本为原始用户查询，属性来自IntentRecognition结果的"属性条件"字段。
          如果"属性条件"为"无"或空，则仅传查询文本，如'狗的图片'。
          如果"属性条件"有值，则附加属性，如'狗的图片|棕色|站立'。
        - 如果意图结果中"需要检索"为"否"，直接结束并回复用户。
        3. 如果工具返回了结果，组织一句话流畅地回答用户。
        '''
        
        # 使用 LangGraph 的 prebuilt 模块创建 React 模式 Agent 
        self.agent_executor = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=system
        )

    def chat(self, user_query: str):
        """主入口"""
        # LangGraph 的输入格式要求是一个包含人类提问的消息列表或者 dict
        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        # 兼容原来的返回数据结构
        final_reply = ""
        
        # 执行 Agent 节点，只打印关键步骤
        for chunk in self.agent_executor.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            msg_type = message.type
            
            # 只打印工具调用和最终结果
            if msg_type == "ai":
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tc in message.tool_calls:
                        print(f"[Agent 调用工具] {tc['name']}")
                else:
                    final_reply = message.content
            elif msg_type == "tool":
                # 工具返回只打印简要信息
                content = str(message.content)[:100]
                print(f"[工具返回] {content}...")
            
        print(f"\n[最终回复] {final_reply}")
        return {"output": final_reply}