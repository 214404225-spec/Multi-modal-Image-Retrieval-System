"""
意图识别输出解析器
负责解析LLM输出的意图识别结果
"""

import re
from typing import Dict, Any

from .constants import CHINESE_NUMBERS


def parse_output(output_text: str, query: str = "") -> Dict[str, Any]:
    """
    基于原代码重构的解析逻辑，融合显示和隐式匹配，增强鲁棒性。
    新增：数量提取功能
    
    Args:
        output_text: LLM输出的原始文本
        query: 原始用户查询（用于补充提取）
        
    Returns:
        解析后的意图字典
    """
    text = str(output_text).strip()
    text_lower = text.lower()
    
    # 1. 判断是否需要检索
    # 注意："否" 不能单独匹配，因为"是否"（whether）是中性的
    is_negative = any(k in text for k in ["不需要", "无需", "无须", "无关", "不用",
                                            "检索：否", "检索:否"])
    is_positive = ("需要" in text or "检索：是" in text or "检索:是" in text) and (not is_negative)
    
    if not is_positive and not is_negative:
        if any(k in text_lower for k in ["top", "阈值", "门槛"]):
            is_positive = True
        else:
            has_retrieval_feat = bool(re.search(r'\d+', text)) or any(k in text for k in ["张", "图", "搜", "找"])
            if has_retrieval_feat:
                is_positive = True
                
    need_retrieval = is_positive
    
    # 2. 提取类别 - 直接从LLM输出中解析
    category = None
    if need_retrieval:
        category_match = re.search(r'检索类别[：:]\s*([^\n]+)', text)
        if category_match:
            category_text = category_match.group(1).strip()
            if category_text != "无":
                category = category_text

        # LLM 未提取到类别时，从原始 query 中尝试提取常见物体
        if category is None:
            common_objects = ["狗", "猫", "熊", "大象", "键盘", "椅子", "桌子",
                            "汽车", "马", "鸟", "鱼", "花", "树", "手机", "电脑",
                            "杯子", "书", "包", "鞋子", "衣服"]
            for obj in common_objects:
                if obj in query:
                    category = obj
                    break
                
    # 3. 提取数量
    count = None
    count_type = None  # "specific" 或 "vague"
    
    if need_retrieval:
        # 3.1 尝试从LLM输出中提取数量
        count_match = re.search(r'检索数量[：:]\s*([^\n]+)', text)
        if count_match:
            count_text = count_match.group(1).strip()
            if count_text == "无":
                count = None
            else:
                # 尝试解析数字
                num_match = re.search(r'(\d+)', count_text)
                if num_match:
                    count = int(num_match.group(1))
                    count_type = "specific"
                else:
                    count = count_text  # 保留模糊描述如"很多"
                    count_type = "vague"
        
        # 3.2 从原始query中提取数量（优先于LLM输出，因为LLM经常误判）
        # 提取query中的具体数字
        query_num_match = re.search(r'(\d+)\s*[张只幅个份条]', query)
        if not query_num_match:
            query_num_match = re.search(r'([一二三四五六七八九十两]+)\s*[张只幅个份条]', query)
        if query_num_match:
            num_val = query_num_match.group(1)
            if num_val.isdigit():
                count = int(num_val)
            else:
                count = CHINESE_NUMBERS.get(num_val, num_val)
            count_type = "specific"

        # 3.3 如果LLM没有输出数量，从LLM输出中提取
        if count is None:
            # 提取阿拉伯数字
            num_match = re.search(r'(\d+)\s*[张只幅个份条]', text)
            if num_match:
                count = int(num_match.group(1))
                count_type = "specific"
            else:
                # 提取中文数字
                cn_num_match = re.search(r'([一二三四五六七八九十两]+)\s*[张只幅个份条]', text)
                if cn_num_match:
                    cn_num = cn_num_match.group(1)
                    count = CHINESE_NUMBERS.get(cn_num, cn_num)
                    count_type = "specific"
                else:
                    # 检查模糊量词
                    vague_words = ["很多", "许多", "一些", "几张", "若干", "大量", "全部", "多张", "几只", "几头"]
                    for word in vague_words:
                        if word in text:
                            count = word
                            count_type = "vague"
                            break

    # 6. 提取物体数量（图片里有几个物体，区别于结果数量"几张"）
    object_count = None
    if need_retrieval:
        object_cls = r'[只个条头匹群双对]'
        pairs = re.findall(r'(\d+)\s*(' + object_cls + r')', query)
        if pairs:
            object_count = int(pairs[-1][0])
        else:
            cn_pairs = re.findall(r'([一二三四五六七八九十两]+)\s*(' + object_cls + r')', query)
            if cn_pairs:
                num = CHINESE_NUMBERS.get(cn_pairs[-1][0])
                if num is not None:
                    object_count = num

    # 4. 检索方式：统一 TopK
    method = "TopK" if need_retrieval else None
    
    # 5. 提取属性条件
    attributes = []
    attr_match = re.search(r'属性条件[：:]\s*([^\n]+)', text)
    if attr_match:
        attr_text = attr_match.group(1).strip()
        if attr_text != "无":
            attributes = [a.strip() for a in attr_text.split(",") if a.strip()]
            
    # 如果LLM没有输出属性，尝试从原始query中提取
    if not attributes and need_retrieval:
        attr_patterns = [
            # 颜色
            "棕色的", "黑色的", "白色的", "黄色的", "灰色的", "红色的",
            "蓝色的", "绿色的",
            # 姿态
            "站立的", "坐着的", "躺着的", "奔跑的", "跳跃的", "飞翔的",
            # 场景
            "室内", "室外", "草地上", "雪地中", "水中", "天空",
            # 构图
            "全身的", "全身", "局部的", "局部", "半身", "特写",
            # 视角
            "正面", "侧面", "背面", "俯视", "仰视",
            # 背景
            "纯色背景", "白色背景", "自然背景",
            # 光照
            "明亮的", "昏暗的", "逆光",
            # 清晰度（保留）
            "清晰的", "模糊的",
        ]
        for pattern in attr_patterns:
            if pattern in query:
                attributes.append(pattern)

        # 单字符属性：需排除为类别词一部分的情况（如"大"是"大象"的一部分）
        if category:
            for word in ["大", "小"]:
                if word in query and word not in category:
                    attributes.append(word)
                
    return {
        "need_retrieval": need_retrieval,
        "category": category,
        "count": count,
        "count_type": count_type,
        "method": method,
        "attributes": attributes,
        "object_count": object_count,
        "raw_output": text
    }