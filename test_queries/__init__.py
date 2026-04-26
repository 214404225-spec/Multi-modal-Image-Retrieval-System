"""
test_queries — 多模态图像检索系统的自动化集成测试

包含 25 条预定义查询，覆盖三条路由路径：
- RegularImageRetrieval（无属性条件，单阶段 CLIP 检索）
- FineGrainedRetrieval（有属性条件，粗排 CLIP + 精排 VL_Refine）
- 直接回复（不需要检索）
"""

from .runner import run_tests

__all__ = ["run_tests"]
