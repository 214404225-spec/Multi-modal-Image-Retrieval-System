"""
细粒度检索模块的VL模型组件
使用Ollama调用VL模型进行属性条件验证（VL_Refine）
"""

import os
import concurrent.futures
from typing import Dict, List

from .constants import VL_OLLAMA_MODEL

# VL 调用超时（秒），单张图像推理通常在 5-15s，设置 60s 兜底
VL_CALL_TIMEOUT = 60
VL_HTTP_TIMEOUT = 50  # HTTP 层超时，必须 < VL_CALL_TIMEOUT 以确保线程不会因 HTTP 请求未完成而被遗弃
VL_MAX_RETRIES = 1  # 超时后重试次数


class VLModelManager:
    """VL模型管理器，通过Ollama调用VL模型"""

    def __init__(self, model_name: str = VL_OLLAMA_MODEL):
        self.model_name = model_name
        self.vl_model = None
        self._init_ollama()

    def _init_ollama(self) -> bool:
        """初始化Ollama VL模型"""
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"[VL模型] Ollama服务未运行")
                return False

            if self.model_name not in result.stdout and self.model_name.split(":")[0] not in result.stdout:
                print(f"[VL模型] 模型 '{self.model_name}' 未安装，请运行: ollama pull {self.model_name}")
                return False

            from langchain_ollama import ChatOllama
            self.vl_model = ChatOllama(
                model=self.model_name, temperature=0.0,
                timeout=VL_HTTP_TIMEOUT,  # 必须 < VL_CALL_TIMEOUT，确保 HTTP 先超时释放线程
            )
            print(f"[VL模型] Ollama 初始化成功（模型: {self.model_name}）")
            return True
        except Exception as e:
            print(f"[VL模型] Ollama初始化失败: {str(e)}")
            return False

    def reset_model(self):
        """重置 VL 模型实例，强制关闭旧 HTTP 连接。超时后调用以避免后续请求排队。"""
        self.vl_model = None
        self._init_ollama()


class VLRefiner:
    """
    VL 精排器：使用 VL 模型对粗排候选图像进行属性条件验证。
    对应 v1.md 架构中的 VL_Refine 组件。
    """

    def __init__(self, vl_manager: "VLModelManager"):
        self.vl_manager = vl_manager

    def refine(self, results: List[Dict], category: str,
               attributes: List[str], top_k: int = None,
               progress_callback=None) -> List[Dict]:
        """
        两阶段检索的精排阶段：VL 二分类验证属性条件（是/否）。
        VL 通过 → 保留 CLIP 粗排分排序；VL 不通过 → 剔除。
        """
        if not attributes or not results:
            return results

        if self.vl_manager.vl_model is None:
            print("[VL_Refine] VL 模型未初始化，跳过精排，返回原始粗排结果")
            return results

        max_refine = len(results)
        candidates = results[:max_refine]
        print(f"[VL_Refine] 开始精排，对 {max_refine} 张候选图像验证属性: {attributes}")

        attr_text = "、".join(attributes)
        passed = []
        for i, r in enumerate(candidates):
            print(f"  [VL_Refine] 验证候选 {i+1}/{max_refine}...")
            if progress_callback:
                progress_callback({
                    "stage": "vl_refine",
                    "current": i + 1,
                    "total": max_refine,
                    "current_image": os.path.basename(r["url"])
                })
            vl_score = self._score_attributes(r["url"], category, attr_text)
            r["vl_score"] = vl_score
            if vl_score > 0:
                r["final_score"] = r.get("score", 0)
                passed.append(r)
            else:
                print(f"    -> 属性不匹配，已剔除")

        # 未进入精排的候选保持原分数
        for r in results[max_refine:]:
            r["vl_score"] = r.get("score", 0)
            r["final_score"] = r.get("score", 0)

        refined = sorted(passed + results[max_refine:],
                        key=lambda x: x.get("final_score", 0), reverse=True)
        print(f"[VL_Refine] 精排完成，通过 {len(passed)}/{max_refine} 张，共 {len(refined)} 张")
        return refined[:top_k] if top_k else refined

    def _score_attributes(self, image_path: str, category: str,
                          attr_text: str) -> float:
        """
        使用 VL 模型对单张图像进行属性匹配评分（带超时和重试）。

        Returns:
            0.0 ~ 1.0 之间的属性匹配分数
        """
        if self.vl_manager.vl_model is None:
            return 0.0

        import base64
        from langchain_core.messages import HumanMessage

        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            ext = image_path.lower().split(".")[-1]
            mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

            prompt = (
                f"请判断这张图片中的{category}是否符合以下属性条件：{attr_text}。\n"
                f"只回复「是」或「否」，不要回复任何其他内容。"
            )
            messages = [
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ])
            ]

            content = self._invoke_vl_with_retry(messages, image_path)
            if content is None:
                return 0.0

            content = content.strip()
            print(f"    [VL raw] {content}")

            if "是" in content and "否" not in content:
                return 1.0
            return 0.0
        except Exception as e:
            print(f"[VLRefiner] 评分失败 [{os.path.basename(image_path)}]: {str(e)}")
            return 0.0

    def _invoke_vl_with_retry(self, messages, image_path: str):
        """带超时和重试的 VL 模型调用。成功返回 content，失败返回 None。

        超时后重置模型实例以强制关闭底层 HTTP 连接，避免后续请求
        因 Ollama 仍在处理已放弃的旧请求而排队卡死。
        """
        last_error = None
        for attempt in range(VL_MAX_RETRIES + 1):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self.vl_manager.vl_model.invoke, messages)
                response = future.result(timeout=VL_CALL_TIMEOUT)
                executor.shutdown(wait=False)
                return response.content if hasattr(response, 'content') else str(response)
            except concurrent.futures.TimeoutError:
                last_error = f"超时 ({VL_CALL_TIMEOUT}s)"
                executor.shutdown(wait=False)
                self.vl_manager.reset_model()  # 强制关闭旧连接，避免后续请求排队
                if attempt < VL_MAX_RETRIES:
                    print(f"    [VL超时] 重试 {attempt + 1}/{VL_MAX_RETRIES}...")
            except Exception as e:
                last_error = str(e)
                executor.shutdown(wait=False)
                if attempt < VL_MAX_RETRIES:
                    print(f"    [VL异常] 重试 {attempt + 1}/{VL_MAX_RETRIES}: {e}")

        fname = os.path.basename(image_path)
        print(f"    [VL放弃] {fname}: {last_error}")
        return None