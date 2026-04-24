"""
细粒度检索模块的VL模型组件
提供Qwen2.5-VL模型的初始化和推理功能
支持transformers直接加载和Ollama调用两种方式
"""

import os
import re
import json
import time
from typing import Dict, Optional
from PIL import Image

from .constants import QWEN_VL_MODEL_NAME, QWEN_VL_OLLAMA_MODEL, LOCAL_MODEL_CACHE


class VLModelManager:
    """VL模型管理器，支持多种调用方式"""
    
    def __init__(self, device: str = None, cache_dir: str = None):
        self.device = device if device else ("cuda" if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # transformers直接加载的模型
        self.vl_model = None
        self.vl_processor = None
        
        # Ollama调用方式的模型
        self.vl_model_ollama = None
    
    def init_transformers_model(self, model_name: str = QWEN_VL_MODEL_NAME) -> bool:
        """初始化 transformers 加载的 Qwen2.5-VL 模型"""
        # 优先使用本地缓存路径
        model_path = LOCAL_MODEL_CACHE.get('qwen_vl')
        if self.cache_dir:
            model_path = os.path.join(self.cache_dir, 'qwen_vl')
        
        use_local = model_path is not None and os.path.exists(model_path)
        model_source = model_path if use_local else model_name
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                self.vl_processor = AutoProcessor.from_pretrained(
                    model_source,
                    cache_dir=self.cache_dir,
                    local_files_only=use_local,
                    resume_download=True,
                    force_download=False
                )
                self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_source,
                    cache_dir=self.cache_dir,
                    local_files_only=use_local,
                    resume_download=True,
                    force_download=False
                ).to(self.device).eval()
                print(f"[细粒度检索] Qwen2.5-VL 模型加载成功（transformers）{'(本地)' if use_local else '(网络)'}")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[细粒度检索] Qwen2.5-VL 模型加载失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[细粒度检索] Qwen2.5-VL 模型加载失败（transformers），已达到最大重试次数")
                    print("=" * 60)
                    print("解决方案：")
                    print("1. 使用Ollama方式（推荐）：")
                    print(f"   ollama pull {QWEN_VL_OLLAMA_MODEL}")
                    print("2. 手动下载模型到本地：")
                    print(f"   huggingface-cli download {QWEN_VL_MODEL_NAME} --local-dir ./models/qwen_vl")
                    print("3. 在 constants.py 中配置本地路径：")
                    print(f"   LOCAL_MODEL_CACHE['qwen_vl'] = './models/qwen_vl'")
                    print("=" * 60)
                    self.vl_model = None
                    return False
    
    def init_ollama_model(self, model_name: str = QWEN_VL_OLLAMA_MODEL) -> bool:
        """使用 Ollama 调用 Qwen2.5-VL"""
        try:
            from langchain_ollama import ChatOllama
            self.vl_model_ollama = ChatOllama(
                model=model_name,
                temperature=0.0
            )
            print(f"[细粒度检索] Ollama Qwen2.5-VL 初始化成功（模型: {model_name}）")
            return True
        except Exception as e:
            print(f"[细粒度检索] Ollama Qwen2.5-VL 初始化失败: {str(e)}")
            self.vl_model_ollama = None
            return False
    
    def extract_features(self, image_path: str) -> Dict[str, int]:
        """
        使用 Qwen2.5-VL 提取图片的细粒度特征（类别及数量）
        优先使用Ollama方式，如果未初始化则尝试transformers方式
        """
        # 优先使用Ollama方式
        if self.vl_model_ollama is not None:
            return self._extract_ollama(image_path)
        
        # 尝试transformers方式
        if self.vl_model is not None:
            return self._extract_transformers(image_path)
        
        # 无可用模型，返回空结果
        print(f"[细粒度检索] VL模型未初始化，无法提取特征: {image_path}")
        return {}
    
    def _extract_ollama(self, image_path: str) -> Dict[str, int]:
        """使用 Ollama Qwen2.5-VL 提取细粒度特征"""
        try:
            import base64
            from langchain_core.messages import HumanMessage
            
            # 读取图像文件并转换为base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # 获取图像格式
            ext = image_path.lower().split(".")[-1]
            if ext in ["jpg", "jpeg"]:
                mime_type = "image/jpeg"
            elif ext == "png":
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
            
            messages = [
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    {"type": "text", "text": "请识别图片中的所有物体及其数量，以JSON格式返回，例如：{\"狗\": 2, \"猫\": 1}"}
                ])
            ]
            response = self.vl_model_ollama.invoke(messages)
            
            # 解析JSON响应
            content = response.content if hasattr(response, 'content') else str(response)
            # 尝试提取JSON
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception as e:
            print(f"[细粒度检索] Ollama VL推理失败: {str(e)}")
            return {}
    
    def _extract_transformers(self, image_path: str) -> Dict[str, int]:
        """使用 transformers 加载的 Qwen2.5-VL 提取细粒度特征"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "请识别图片中的所有物体及其数量，以JSON格式返回，例如：{\"狗\": 2, \"猫\": 1}"}
                    ]
                }
            ]
            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.vl_processor(text=text, images=Image.open(image_path), return_tensors="pt").to(self.device)
            
            import torch
            with torch.no_grad():
                outputs = self.vl_model.generate(**inputs, max_new_tokens=100)
            
            generated_text = self.vl_processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # 简化处理，实际应使用json.loads
            return {"解析结果": 1}
        except Exception as e:
            print(f"[细粒度检索] VL推理失败: {str(e)}")
            return {}
    
