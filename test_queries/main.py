"""
测试脚本主入口
运行25个测试用例
"""

from .runner import run_tests


if __name__ == "__main__":
    results = run_tests()
    print(f"\n测试完成！共运行 {len(results)} 个测试用例。")