"""
核心纯函数逻辑的单元测试（不加载模型、不依赖 Ollama，可快速运行）。

运行方式：
    python -m pytest tests/ -v
"""

import pytest

from agent_pipeline.pipeline import MultiModalAgentPipeline
from fine_grained_retrieval_module.vl_models import parse_vl_yes_no


# ── VL 是/否 回复解析 ──

class TestParseVlYesNo:
    @pytest.mark.parametrize("content", ["是", "是的", "是。", " 是 ", "是的，符合"])
    def test_affirmative(self, content):
        assert parse_vl_yes_no(content) is True

    @pytest.mark.parametrize("content", ["否", "不是", "不是的", "不是。", "并不是", "不符合"])
    def test_negative(self, content):
        """关键回归：'不是' 含 '是' 字，旧逻辑会误判为通过"""
        assert parse_vl_yes_no(content) is False

    @pytest.mark.parametrize("content", ["", "   ", "无法判断", "图中没有狗"])
    def test_ambiguous_defaults_to_reject(self, content):
        assert parse_vl_yes_no(content) is False


# ── CLIP 友好属性判定 ──

class TestIsClipFriendly:
    @pytest.mark.parametrize("attr", ["白色", "粉色", "紫色", "橙色", "藏青色"])
    def test_colors_end_with_se(self, attr):
        """以'色'结尾的颜色词全部归为简单属性（模式匹配而非白名单）"""
        assert MultiModalAgentPipeline._is_clip_friendly(attr) is True

    @pytest.mark.parametrize("attr", ["红", "白", "黑", "金", "粉"])
    def test_single_char_colors(self, attr):
        assert MultiModalAgentPipeline._is_clip_friendly(attr) is True

    @pytest.mark.parametrize("attr", ["大", "小", "巨大", "明亮", "昏暗", "暗"])
    def test_size_and_lighting(self, attr):
        assert MultiModalAgentPipeline._is_clip_friendly(attr) is True

    @pytest.mark.parametrize("attr", ["站立", "坐着", "室内", "正面", "全身", "纯色背景"])
    def test_complex_attrs_not_clip_friendly(self, attr):
        """姿态/场景/视角/构图/背景类复杂属性必须走 VL 精排"""
        assert MultiModalAgentPipeline._is_clip_friendly(attr) is False


# ── 检索短语构建 ──

class TestBuildSearchQuery:
    def test_arabic_number_with_classifier(self):
        assert MultiModalAgentPipeline._build_search_query(
            "帮我找3张3只狗的照片", "狗") == "3只狗"

    def test_chinese_number_with_classifier(self):
        assert MultiModalAgentPipeline._build_search_query(
            "找两只猫的图片", "猫") == "两只猫"

    def test_no_number_falls_back_to_category(self):
        assert MultiModalAgentPipeline._build_search_query(
            "帮我找狗的照片", "狗") == "狗"

    def test_empty_category_returns_query(self):
        assert MultiModalAgentPipeline._build_search_query(
            "随便找点图片", "") == "随便找点图片"


# ── 数量提取与 TopK 解析 ──

class TestExtractCount:
    def test_arabic_number(self):
        assert MultiModalAgentPipeline._extract_count("找3张狗的图片") == 3

    def test_chinese_number(self):
        assert MultiModalAgentPipeline._extract_count("找三张狗的图片") == 3

    def test_fuzzy_many(self):
        assert MultiModalAgentPipeline._extract_count("找很多狗的图片") == 20

    def test_fuzzy_some(self):
        assert MultiModalAgentPipeline._extract_count("找一些狗的图片") == 10

    def test_default(self):
        assert MultiModalAgentPipeline._extract_count("找狗的图片") == 5


class TestResolveTopK:
    def test_int_count(self):
        assert MultiModalAgentPipeline._resolve_top_k(3) == 3

    def test_fuzzy_many(self):
        assert MultiModalAgentPipeline._resolve_top_k("很多") == 20

    def test_fuzzy_some(self):
        assert MultiModalAgentPipeline._resolve_top_k("几张") == 10

    def test_none_default(self):
        assert MultiModalAgentPipeline._resolve_top_k(None) == 5

    def test_zero_falls_back(self):
        assert MultiModalAgentPipeline._resolve_top_k(0) == 5
