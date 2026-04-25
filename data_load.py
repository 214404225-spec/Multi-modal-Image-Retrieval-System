"""
图像数据集加载模块
统一管理图像路径发现、数量统计和采样
"""

import os
import glob
from typing import List, Tuple

# 支持的图像格式扩展名
_IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# 默认图像目录（项目根目录下的 test_images）
# data_load.py 位于项目根目录，直接取同级 test_images
DEFAULT_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")


def discover_image_paths(directory: str = DEFAULT_IMAGE_DIR) -> List[str]:
    """发现并返回指定目录下所有图像文件的绝对路径（已排序去重）"""
    if not os.path.exists(directory):
        print(f"[警告] 图像目录不存在: {directory}")
        return []

    paths = set()
    for ext in _IMAGE_EXTENSIONS:
        paths.update(glob.glob(os.path.join(directory, ext)))

    abs_paths = [os.path.abspath(p) for p in paths]
    abs_paths.sort()
    return abs_paths


def get_image_counts(image_paths: List[str]) -> int:
    """返回可用图像总数"""
    return len(image_paths)


def split_for_indexing(
    image_paths: List[str],
    regular_size: int = -1,
    fine_grained_size: int = 20,
) -> Tuple[List[str], List[str]]:
    """
    根据配置切分两个检索模块的图像采样。

    Args:
        image_paths: 全部图像路径列表
        regular_size: 常规检索模块索引数量，-1 表示全部
        fine_grained_size: 细粒度检索模块索引数量，默认 20

    Returns:
        (regular_paths, fine_grained_paths) 两个采样路径列表
    """
    total = len(image_paths)
    if total == 0:
        return [], []

    if regular_size < 0 or regular_size > total:
        regular_size = total
    if fine_grained_size > total:
        fine_grained_size = total

    regular_paths = image_paths[:regular_size]
    fine_grained_paths = image_paths[:fine_grained_size]
    return regular_paths, fine_grained_paths
