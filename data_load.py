"""
图像数据集加载模块
统一管理图像路径发现、数量统计和采样
"""

import os
import glob
from typing import List

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


