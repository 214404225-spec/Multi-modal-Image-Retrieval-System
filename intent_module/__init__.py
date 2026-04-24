"""
意图识别模块包
提供基于Qwen2.5-3B的用户意图识别功能
"""

from .module import IntentRecognitionModule
from .constants import CHINESE_NUMBERS

__all__ = ["IntentRecognitionModule", "CHINESE_NUMBERS"]
