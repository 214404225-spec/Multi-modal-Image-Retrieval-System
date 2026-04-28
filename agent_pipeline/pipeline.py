"""
Agent Pipeline主类
程序化路由编排（对应 v1.md 架构：意图识别 → 判定是否有属性条件 → 路由）
"""

import os
import re
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from intent_module import IntentRecognitionModule
from intent_module.constants import INTENT_MODEL_NAME
from regular_retrieval_module import RegularRetrievalModule
from regular_retrieval_module.constants import CLIP_MODEL_PATH
from fine_grained_retrieval_module import FineGrainedRetrievalModule
from shared.clip_encoder import CLIPEncoder
from data_load import discover_image_paths, get_image_counts


def _trim_result_paths(res: dict) -> dict:
    """将检索结果中的完整路径截短为文件名"""
    if "results" not in res or not isinstance(res["results"], list):
        return res
    trimmed_results = []
    for r in res["results"]:
        trimmed = dict(r)
        if "url" in trimmed:
            trimmed["url"] = os.path.basename(trimmed["url"])
        trimmed_results.append(trimmed)
    return {**res, "results": trimmed_results}


def _format_response(llm: ChatOllama, user_query: str,
                     retrieval_result: dict) -> str:
    """使用 LLM 将检索结果格式化为用户友好的回复"""
    results = retrieval_result.get("results", [])
    if not results:
        return "未找到匹配的图片。"

    filenames = [r.get("url", "") for r in results]
    file_list = "\n".join(f"- {name}" for name in filenames if name)

    prompt = (
        f"用户查询：{user_query}\n\n"
        f"检索到 {len(filenames)} 张图片：\n{file_list}\n\n"
        f"请用一句话总结结果，然后列出图片文件名。不要编造链接，直接使用上面给出的文件名。"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return f"找到 {len(filenames)} 张图片：\n{file_list}"


class MultiModalAgentPipeline:
    """
    多模态图像检索流水线（v1.md 架构）：

    路由逻辑（程序化，不由 LLM 决定）：
    - 意图识别 → 有属性条件 → 细粒度两阶段检索（粗排 CLIP + 精排 VL_Refine）
    - 意图识别 → 无属性条件 → 常规检索（CLIP 单阶段）
    - 意图识别 → 不需要检索 → 直接回复
    """

    def __init__(self, model_name: str = None, image_dir: str = None):
        # 意图识别/回复格式化所用的 LLM（由 INTENT_MODEL 环境变量或常量默认值控制）
        if model_name is None:
            model_name = INTENT_MODEL_NAME
        print(f"[INFO] 意图 LLM: {model_name}")

        self.llm = ChatOllama(model=model_name, temperature=0.0)
        self.image_dir = image_dir  # 记录当前使用的图像目录

        # --- 1. 初始化各模块 ---
        self.intent_module = IntentRecognitionModule(llm=self.llm)

        shared_clip = CLIPEncoder(clip_vision_path=CLIP_MODEL_PATH)
        self.regular_retrieval = RegularRetrievalModule(clip_encoder=shared_clip)

        self.fine_grained = FineGrainedRetrievalModule(
            clip_encoder=shared_clip,
            offline_indexer=self.regular_retrieval.offline_indexer
        )

        # --- 2. 加载图像并建索引 ---
        all_image_paths = discover_image_paths(image_dir) if image_dir else discover_image_paths()
        total_count = get_image_counts(all_image_paths)
        print(f"[INFO] 图像数据集共 {total_count} 张")

        try:
            regular_index_size = int(os.environ.get("REGULAR_INDEX_SIZE", "-1"))
        except ValueError:
            print(f"[警告] REGULAR_INDEX_SIZE 值无效，使用默认值 -1（全部图像）")
            regular_index_size = -1

        regular_paths = all_image_paths if regular_index_size < 0 else all_image_paths[:regular_index_size]

        if regular_paths:
            print(f"[INFO] 开始离线索引（共{len(regular_paths)}张）...")
            self.regular_retrieval.offline_indexing(regular_paths)
        else:
            print(f"[警告] 未找到图像，检索功能将不可用")

    def chat(self, user_query: str) -> dict:
        """
        主入口：程序化路由（对应 v1.md）

        1. 意图识别 → 2. 判定路由 → 3. 检索 → 4. 格式化回复
        """
        # Step 1: 意图识别
        intent = self.intent_module.analyze_intent(user_query)
        print(f"[意图] {intent.get('need_retrieval')} | "
              f"类别={intent.get('category')} | "
              f"属性={intent.get('attributes')} | "
              f"方式={intent.get('method')} | "
              f"数量={intent.get('count')}")

        need_retrieval = intent.get("need_retrieval", False)
        attributes = intent.get("attributes", [])
        category = intent.get("category", "")
        method = intent.get("method", "TopK")
        count = intent.get("count")

        # Step 2: 程序化路由（不由 LLM agent 决定）
        if not need_retrieval:
            return {"output": "不需要检索。"}

        if not category:
            # 意图模块未提取到类别，直接用原始查询做常规检索
            print(f"[路由] 常规检索（无类别，用原始查询）")
            k = self._extract_count(user_query)
            res = self.regular_retrieval.retrieve(user_query, method="TopK", top_k=k)
            res = _trim_result_paths(res)
            output = _format_response(self.llm, user_query, res)
            return {"output": output}

        if attributes:
            # 有属性条件 → 细粒度两阶段检索
            print(f"[路由] 细粒度检索: {category}, 属性={attributes}, {method}")
            # 粗排取 count*3 候选（保底 15），留足余量供 VL 精排筛选
            coarse_k = max((count or 5) * 3, 15) if isinstance(count, int) else 30
            res = self.fine_grained.online_retrieval(
                category=category,
                method=method or "TopK",
                target_count=count if isinstance(count, int) else None,
                top_k=coarse_k,
                attributes=attributes
            )
            if "results" in res:
                res["results"] = self.fine_grained.refine_by_attributes(
                    res.get("results", []), category, attributes
                )
                # TopK + 明确数量时，精排后裁剪到精确数量
                if method == "TopK" and isinstance(count, int) and count > 0:
                    if len(res["results"]) > count:
                        print(f"[路由] 精排后裁剪 {len(res['results'])} → {count} 张")
                    res["results"] = res["results"][:count]
                res["refinement_applied"] = True
                res["refinement_method"] = "VL_Refine"
            res = _trim_result_paths(res)
            output = _format_response(self.llm, user_query, res)
            return {"output": output}
        else:
            # 无属性条件 → 常规检索
            print(f"[路由] 常规检索: {category}, {method}")
            k = self._resolve_top_k(method, count)
            res = self.regular_retrieval.retrieve(
                query=category, method=method or "TopK", top_k=k
            )
            res = _trim_result_paths(res)
            output = _format_response(self.llm, user_query, res)
            return {"output": output}

    def chat_structured(self, user_query: str, progress_callback=None) -> dict:
        """
        与 chat() 相同的路由逻辑，但返回结构化数据供 Web 前端使用。

        Returns:
            {"output": "<formatted string>", "structured": {...}}
        """
        # Step 1: 意图识别
        self._report_progress(progress_callback, "intent", "正在分析意图...")
        intent = self.intent_module.analyze_intent(user_query)
        print(f"[意图] {intent.get('need_retrieval')} | "
              f"类别={intent.get('category')} | "
              f"属性={intent.get('attributes')} | "
              f"方式={intent.get('method')} | "
              f"数量={intent.get('count')}")

        need_retrieval = intent.get("need_retrieval", False)
        attributes = intent.get("attributes", [])
        category = intent.get("category", "")
        method = intent.get("method", "TopK")
        count = intent.get("count")

        if not need_retrieval:
            return {
                "output": "不需要检索。",
                "structured": {"route": "no_retrieval", "query": user_query,
                               "intent": intent, "results": [], "total_results": 0}
            }

        if not category:
            print(f"[路由] 常规检索（无类别，用原始查询）")
            self._report_progress(progress_callback, "retrieval", "正在进行常规检索...")
            k = self._extract_count(user_query)
            res = self.regular_retrieval.retrieve(user_query, method="TopK", top_k=k)
            res = _trim_result_paths(res)
            self._report_progress(progress_callback, "formatting", "正在生成回复...")
            output = _format_response(self.llm, user_query, res)
            return {
                "output": output,
                "structured": {
                    "route": "regular",
                    "query": user_query,
                    "intent": intent,
                    "results": res.get("results", []),
                    "total_results": len(res.get("results", []))
                }
            }

        if attributes:
            print(f"[路由] 细粒度检索: {category}, 属性={attributes}, {method}")
            self._report_progress(progress_callback, "retrieval", "正在进行粗排检索...")
            coarse_k = max((count or 5) * 3, 15) if isinstance(count, int) else 30
            res = self.fine_grained.online_retrieval(
                category=category,
                method=method or "TopK",
                target_count=count if isinstance(count, int) else None,
                top_k=coarse_k,
                attributes=attributes
            )
            if "results" in res:
                total_candidates = len(res.get("results", []))
                self._report_progress(progress_callback, "vl_refine",
                                      f"正在进行VL精排验证 ({total_candidates} 张候选)...",
                                      current=0, total=total_candidates)
                res["results"] = self.fine_grained.refine_by_attributes(
                    res.get("results", []), category, attributes,
                    progress_callback=progress_callback
                )
                if method == "TopK" and isinstance(count, int) and count > 0:
                    if len(res["results"]) > count:
                        print(f"[路由] 精排后裁剪 {len(res['results'])} → {count} 张")
                    res["results"] = res["results"][:count]
                res["refinement_applied"] = True
                res["refinement_method"] = "VL_Refine"
            res = _trim_result_paths(res)
            self._report_progress(progress_callback, "formatting", "正在生成回复...")
            output = _format_response(self.llm, user_query, res)
            return {
                "output": output,
                "structured": {
                    "route": "fine_grained",
                    "query": user_query,
                    "intent": intent,
                    "results": res.get("results", []),
                    "total_results": len(res.get("results", [])),
                    "refinement_applied": True,
                    "refinement_method": "VL_Refine"
                }
            }
        else:
            print(f"[路由] 常规检索: {category}, {method}")
            self._report_progress(progress_callback, "retrieval", "正在进行常规检索...")
            k = self._resolve_top_k(method, count)
            res = self.regular_retrieval.retrieve(
                query=category, method=method or "TopK", top_k=k
            )
            res = _trim_result_paths(res)
            self._report_progress(progress_callback, "formatting", "正在生成回复...")
            output = _format_response(self.llm, user_query, res)
            return {
                "output": output,
                "structured": {
                    "route": "regular",
                    "query": user_query,
                    "intent": intent,
                    "results": res.get("results", []),
                    "total_results": len(res.get("results", []))
                }
            }

    @staticmethod
    def _report_progress(callback, stage: str, message: str, **kwargs):
        """向 progress_callback 报告进度（如果已设置）"""
        if callback:
            callback({"stage": stage, "message": message, **kwargs})

    @staticmethod
    def _extract_count(query: str) -> int:
        """从查询文本中提取数量，默认返回 5"""
        m = re.search(r'(\d+)\s*[张只幅个份条]', query)
        if m:
            return int(m.group(1))
        cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "两": 2}
        m = re.search(r'([一二三四五六七八九十两]+)\s*[张只幅个份条]', query)
        if m:
            return cn_map.get(m.group(1), 5)
        return 5

    @staticmethod
    def _resolve_top_k(method: str, count) -> int:
        """根据检索方式和数量确定 Top-K 值"""
        if method == "TopK" and isinstance(count, int) and count > 0:
            return count
        return 5
