"""
细粒度检索模块的VL模型组件
使用Ollama调用VL模型进行属性条件验证（VL_Refine）
"""

import os
import concurrent.futures
from typing import Dict, List

from .constants import VL_OLLAMA_MODEL, VL_PARALLEL_WORKERS

VL_HTTP_TIMEOUT = 50  # ChatOllama httpx 超时（秒），单张图像推理通常在 5-15s
VL_MAX_RETRIES = 1  # 超时/异常后重试次数


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
                timeout=VL_HTTP_TIMEOUT,
            )
            print(f"[VL模型] Ollama 初始化成功（模型: {self.model_name}）")
            return True
        except Exception as e:
            print(f"[VL模型] Ollama初始化失败: {str(e)}")
            return False

    def create_chat_model(self):
        """创建独立的 ChatOllama 实例（轻量级，用于并行验证）。"""
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=self.model_name, temperature=0.0,
            timeout=VL_HTTP_TIMEOUT,
        )

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
               attributes: List[str] = None, object_count: int = None,
               top_k: int = None,
               progress_callback=None,
               max_workers: int = None) -> List[Dict]:
        """
        VL 精排：对粗排候选图像验证属性条件 / 物体数量（可合并）。
        VL 通过 → 保留 CLIP 粗排分排序；VL 不通过 → 剔除。

        max_workers: 并行验证的 worker 数（默认读取 VL_PARALLEL_WORKERS，1 即串行）。
        """
        if not results:
            return results

        if self.vl_manager.vl_model is None:
            print("[VL_Refine] VL 模型未初始化，跳过精排，返回原始粗排结果")
            return results

        if max_workers is None:
            max_workers = VL_PARALLEL_WORKERS

        max_refine = len(results)
        candidates = results[:max_refine]

        # 构建日志描述
        parts = []
        if object_count:
            parts.append(f"{object_count}个{category}")
        if attributes:
            parts.append(f"属性={attributes}")
        desc = ", ".join(parts) if parts else category

        if max_workers > 1 and len(candidates) > 1:
            mode = f"并行({max_workers} workers)"
        else:
            mode = "串行"
        print(f"[VL_Refine] 开始精排({mode})，对 {max_refine} 张候选验证: {desc}")

        if max_workers <= 1 or len(candidates) <= 1:
            # 串行路径
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
                vl_score = self._score_attributes(
                    r["url"], category, attributes, object_count
                )
                r["vl_score"] = vl_score
                if vl_score > 0:
                    r["final_score"] = r.get("score", 0)
                    passed.append(r)
                else:
                    print(f"    -> 不匹配，已剔除")
        else:
            # 并行路径：每个 worker 使用独立的 ChatOllama 实例
            passed = []
            completed = 0

            def verify_single(candidate):
                url = candidate["url"]
                chat_model = self.vl_manager.create_chat_model()
                vl_score = self._score_attributes(
                    url, category, attributes, object_count, chat_model=chat_model
                )
                candidate["vl_score"] = vl_score
                if vl_score > 0:
                    candidate["final_score"] = candidate.get("score", 0)
                return candidate, vl_score

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(verify_single, r): r for r in candidates}
                for future in concurrent.futures.as_completed(futures):
                    candidate, vl_score = future.result()
                    completed += 1
                    if progress_callback:
                        progress_callback({
                            "stage": "vl_refine",
                            "current": completed,
                            "total": max_refine,
                            "current_image": os.path.basename(candidate["url"])
                        })
                    if vl_score > 0:
                        passed.append(candidate)
                    else:
                        print(f"  [VL_Refine] 候选不匹配，已剔除")

            # 并行完成后的结果按分数排序
            passed.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        for r in results[max_refine:]:
            r["vl_score"] = r.get("score", 0)
            r["final_score"] = r.get("score", 0)

        refined = sorted(passed + results[max_refine:],
                        key=lambda x: x.get("final_score", 0), reverse=True)
        print(f"[VL_Refine] 精排完成，通过 {len(passed)}/{max_refine} 张，共 {len(refined)} 张")
        return refined[:top_k] if top_k else refined

    def _score_with_prompt(self, image_path: str, prompt: str, chat_model=None) -> float:
        """
        核心 VL 评分：编码图像 → 调用 VL → 解析是/否响应。
        返回 1.0（通过）或 0.0（不通过/失败）。
        chat_model: 可选的外部 ChatOllama 实例（并行模式下使用，避免共享实例）。
        """
        vl_model = chat_model if chat_model is not None else self.vl_manager.vl_model
        if vl_model is None:
            return 0.0

        import base64
        from langchain_core.messages import HumanMessage

        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            ext = image_path.lower().split(".")[-1]
            mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

            messages = [
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ])
            ]

            content = self._invoke_vl_with_retry(messages, image_path, chat_model=chat_model)
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

    def _score_attributes(self, image_path: str, category: str,
                          attributes: list = None,
                          object_count: int = None,
                          chat_model=None) -> float:
        """使用 VL 模型对单张图像进行属性 / 物体数量验证（可合并）。
        chat_model: 可选的外部 ChatOllama 实例（并行模式下使用）。
        """
        if object_count and attributes:
            attr_text = "".join(attributes)
            prompt = (
                f"请判断这张图片中是否恰好有{object_count}个"
                f"{attr_text}的{category}。\n"
                f"只回复「是」或「否」，不要回复任何其他内容。"
            )
        elif object_count:
            prompt = (
                f"请判断这张图片中是否恰好有{object_count}个"
                f"{category}。\n"
                f"只回复「是」或「否」，不要回复任何其他内容。"
            )
        else:
            attr_text = "、".join(attributes) if attributes else ""
            prompt = (
                f"请判断这张图片中的{category}是否符合以下属性条件：{attr_text}。\n"
                f"只回复「是」或「否」，不要回复任何其他内容。"
            )
        return self._score_with_prompt(image_path, prompt, chat_model=chat_model)

    def _invoke_vl_with_retry(self, messages, image_path: str, chat_model=None):
        """带重试的 VL 模型调用。成功返回 content，失败返回 None。

        ChatOllama 自带 httpx 超时（VL_HTTP_TIMEOUT），无需额外包装。
        chat_model: 可选的外部 ChatOllama 实例（并行模式下使用）。
        """
        is_private = chat_model is not None
        vl_model = chat_model if is_private else self.vl_manager.vl_model

        last_error = None
        for attempt in range(VL_MAX_RETRIES + 1):
            try:
                response = vl_model.invoke(messages)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                last_error = str(e)
                if not is_private:
                    self.vl_manager.reset_model()
                    vl_model = self.vl_manager.vl_model
                if attempt < VL_MAX_RETRIES:
                    print(f"    [VL异常] 重试 {attempt + 1}/{VL_MAX_RETRIES}: {e}")

        fname = os.path.basename(image_path)
        print(f"    [VL放弃] {fname}: {last_error}")
        return None