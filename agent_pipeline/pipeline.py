"""
Agent Pipeline主类
程序化路由编排（对应 v1.md 架构：意图识别 → 判定是否有属性条件 → 路由）
"""

import os
import re
import time
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


def _format_response(user_query: str, retrieval_result: dict) -> str:
    """将检索结果格式化为用户友好的回复（纯模板，不调用LLM）"""
    results = retrieval_result.get("results", [])
    if not results:
        return "未找到匹配的图片。"

    filenames = [r.get("url", "") for r in results]
    file_list = "\n".join(f"- {name}" for name in filenames if name)
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

        self.llm = ChatOllama(model=model_name, temperature=0.0, timeout=120)
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
        t_start = time.time()

        # Step 1: 意图识别
        t0 = time.time()
        intent = self.intent_module.analyze_intent(user_query)
        t_intent = time.time() - t0
        print(f"[意图] {intent.get('need_retrieval')} | "
              f"类别={intent.get('category')} | "
              f"属性={intent.get('attributes')} | "
              f"方式={intent.get('method')} | "
              f"数量={intent.get('count')} | "
              f"耗时={t_intent:.1f}s")

        need_retrieval = intent.get("need_retrieval", False)
        attributes = intent.get("attributes", [])
        category = intent.get("category", "")
        count = intent.get("count")
        object_count = intent.get("object_count")

        # Step 2: 程序化路由（不由 LLM agent 决定）
        if not need_retrieval:
            return {"output": "不需要检索。"}

        if not category:
            print(f"[路由] 常规检索（无类别，用原始查询）")
            k = self._extract_count(user_query)
            res = self.regular_retrieval.retrieve(user_query, top_k=k)
            res = _trim_result_paths(res)
            output = _format_response(user_query, res)
            return {"output": output}

        # 判断是否需要 VL 精排：物体数量 或 复杂属性
        complex_attrs = [a for a in attributes if not self._is_clip_friendly(a)]
        needs_vl = bool(object_count) or bool(complex_attrs)

        if needs_vl:
            if object_count:
                base_k = count if isinstance(count, int) else 5
                coarse_k = max(base_k * object_count * 2, 30)
            else:
                coarse_k = max((count or 5) * 3, 15) if isinstance(count, int) else 30
            print(f"[路由] VL精排: 类别={category}, "
                  f"计数={object_count or '无'}, 属性={complex_attrs or '无'}")
            t1 = time.time()
            res = self.fine_grained.online_retrieval(
                category=category, top_k=coarse_k,
            )
            t_retrieval = time.time() - t1
            t_vl = 0
            if "results" in res and res["results"]:
                t2 = time.time()
                res["results"] = self.fine_grained.refine_by_attributes(
                    res["results"], category,
                    attributes=complex_attrs if complex_attrs else None,
                    object_count=object_count,
                )
                t_vl = time.time() - t2
                if isinstance(count, int) and count > 0:
                    if len(res["results"]) > count:
                        print(f"[路由] 精排后裁剪 {len(res['results'])} → {count} 张")
                    res["results"] = res["results"][:count]
                res["refinement_applied"] = True
                res["refinement_method"] = "VL_Refine"
            res = _trim_result_paths(res)
            output = _format_response(user_query, res)
            t_total = time.time() - t_start
            print(f"[计时] 意图={t_intent:.1f}s 检索={t_retrieval:.1f}s VL={t_vl:.1f}s 总计={t_total:.1f}s")
            return {"output": output}

        # CLIP-only：编码全短语直接检索
        search_query = self._build_search_query(user_query, category)
        if attributes:
            search_query = "".join(attributes) + "的" + search_query
        print(f"[路由] 常规检索: {search_query}")
        k = self._resolve_top_k(count)
        t1 = time.time()
        res = self.regular_retrieval.retrieve(
            query=search_query, top_k=k
        )
        t_retrieval = time.time() - t1
        res = _trim_result_paths(res)
        output = _format_response(user_query, res)
        t_total = time.time() - t_start
        print(f"[计时] 意图={t_intent:.1f}s 检索={t_retrieval:.1f}s 总计={t_total:.1f}s")
        return {"output": output}

    def chat_structured(self, user_query: str, progress_callback=None) -> dict:
        """
        与 chat() 相同的路由逻辑，但返回结构化数据供 Web 前端使用。

        Returns:
            {"output": "<formatted string>", "structured": {...}}
        """
        t_start = time.time()

        # Step 1: 意图识别
        self._report_progress(progress_callback, "intent", "正在分析意图...")
        t0 = time.time()
        intent = self.intent_module.analyze_intent(user_query)
        t_intent = time.time() - t0
        print(f"[意图] {intent.get('need_retrieval')} | "
              f"类别={intent.get('category')} | "
              f"属性={intent.get('attributes')} | "
              f"方式={intent.get('method')} | "
              f"数量={intent.get('count')} | "
              f"耗时={t_intent:.1f}s")

        need_retrieval = intent.get("need_retrieval", False)
        attributes = intent.get("attributes", [])
        category = intent.get("category", "")
        count = intent.get("count")
        object_count = intent.get("object_count")

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
            t1 = time.time()
            res = self.regular_retrieval.retrieve(user_query, top_k=k)
            t_retrieval = time.time() - t1
            res = _trim_result_paths(res)
            self._report_progress(progress_callback, "formatting", "正在生成回复...")
            output = _format_response(user_query, res)
            t_total = time.time() - t_start
            print(f"[计时] 意图={t_intent:.1f}s 检索={t_retrieval:.1f}s 总计={t_total:.1f}s")
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

        # 判断是否需要 VL 精排：物体数量 或 复杂属性
        complex_attrs = [a for a in attributes if not self._is_clip_friendly(a)]
        needs_vl = bool(object_count) or bool(complex_attrs)

        if needs_vl:
            if object_count:
                base_k = count if isinstance(count, int) else 5
                coarse_k = max(base_k * object_count * 2, 30)
            else:
                coarse_k = max((count or 5) * 3, 15) if isinstance(count, int) else 30
            print(f"[路由] VL精排: 类别={category}, "
                  f"计数={object_count or '无'}, 属性={complex_attrs or '无'}")
            self._report_progress(progress_callback, "retrieval", "正在进行CLIP粗排检索...")
            t1 = time.time()
            res = self.fine_grained.online_retrieval(
                category=category, top_k=coarse_k,
            )
            t_retrieval = time.time() - t1
            t_vl = 0
            if "results" in res and res["results"]:
                total_candidates = len(res.get("results", []))
                self._report_progress(progress_callback, "vl_refine",
                                      f"正在进行VL精排验证 ({total_candidates} 张候选)...",
                                      current=0, total=total_candidates)
                t2 = time.time()
                res["results"] = self.fine_grained.refine_by_attributes(
                    res["results"], category,
                    attributes=complex_attrs if complex_attrs else None,
                    object_count=object_count,
                    progress_callback=progress_callback,
                )
                t_vl = time.time() - t2
                if isinstance(count, int) and count > 0:
                    if len(res["results"]) > count:
                        print(f"[路由] 精排后裁剪 {len(res['results'])} → {count} 张")
                    res["results"] = res["results"][:count]
                res["refinement_applied"] = True
                res["refinement_method"] = "VL_Refine"
            res = _trim_result_paths(res)
            self._report_progress(progress_callback, "formatting", "正在生成回复...")
            output = _format_response(user_query, res)
            t_total = time.time() - t_start
            print(f"[计时] 意图={t_intent:.1f}s 检索={t_retrieval:.1f}s VL={t_vl:.1f}s 总计={t_total:.1f}s")
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

        # CLIP-only：编码全短语直接检索
        search_query = self._build_search_query(user_query, category)
        if attributes:
            search_query = "".join(attributes) + "的" + search_query
        print(f"[路由] 常规检索: {search_query}")
        self._report_progress(progress_callback, "retrieval", "正在进行CLIP检索...")
        k = self._resolve_top_k(count)
        t1 = time.time()
        res = self.regular_retrieval.retrieve(
            query=search_query, top_k=k
        )
        t_retrieval = time.time() - t1
        res = _trim_result_paths(res)
        self._report_progress(progress_callback, "formatting", "正在生成回复...")
        output = _format_response(user_query, res)
        t_total = time.time() - t_start
        print(f"[计时] 意图={t_intent:.1f}s 检索={t_retrieval:.1f}s 总计={t_total:.1f}s")
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

    # CLIP 能可靠处理的简单属性（颜色、大小、明暗），无需 VL 验证
    @staticmethod
    def _is_clip_friendly(attr: str) -> bool:
        """判断属性是否为简单属性（CLIP 可直接编码，无需 VL 验证）。

        采用模式匹配 + 补充白名单，而非枚举具体取值：
        - 颜色类：以"色"结尾（覆盖"粉色""紫色""橙色"等），+ 常见单字颜色词
        - 大小类：大/小/巨/微
        - 明暗类：明亮/昏暗/暗/亮
        """
        # 颜色：以"色"结尾
        if attr.endswith("色"):
            return True
        # 颜色：常见单字颜色词 + 金属色
        if attr in {"金", "银", "粉", "红", "蓝", "绿", "黄", "白", "黑", "灰",
                     "棕", "紫", "橙", "青", "褐"}:
            return True
        # 大小类
        if attr in {"大", "小", "巨", "微", "巨大", "微小"}:
            return True
        # 明暗类
        if attr in {"明亮", "昏暗", "暗", "亮"}:
            return True
        return False

    @staticmethod
    def _report_progress(callback, stage: str, message: str, **kwargs):
        """向 progress_callback 报告进度（如果已设置）"""
        if callback:
            callback({"stage": stage, "message": message, **kwargs})

    @staticmethod
    def _build_search_query(query: str, category: str) -> str:
        """从原始查询中提取完整检索短语（数量+量词+类别），用于 CLIP 编码。

        例如："帮我找3张3只狗的照片" + category="狗" → "3只狗"
        没有数量修饰时回退到裸类别。
        """
        if not category:
            return query
        classifiers = r'[只个条头匹幅本件辆艘架株朵颗根块片座台把支位名双对群]'
        pattern = rf'(\d+\s*{classifiers}?\s*{re.escape(category)})'
        m = re.search(pattern, query)
        if m:
            return m.group(1)
        # 中文数字
        cn_num = r'[一二三四五六七八九十两]'
        pattern_cn = rf'({cn_num}\s*{classifiers}?\s*{re.escape(category)})'
        m = re.search(pattern_cn, query)
        if m:
            return m.group(1)
        return category

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
        if any(w in query for w in ["很多", "许多", "大量", "全部", "所有"]):
            return 20
        if any(w in query for w in ["一些", "几张", "几只", "几头", "若干"]):
            return 10
        return 5

    @staticmethod
    def _resolve_top_k(count) -> int:
        """根据数量确定 Top-K 值。支持具体数字和模糊量词，默认为 5。"""
        if isinstance(count, int) and count > 0:
            return count
        if isinstance(count, str):
            if count in {"很多", "许多", "大量", "全部", "所有", "多张", "多只", "多个"}:
                return 20
            if count in {"一些", "几张", "几只", "几头", "若干", "几个"}:
                return 10
        return 5
