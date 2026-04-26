"""
自动化测试入口

用法:
    python -m test_queries.main          # 从项目根目录运行
    python test_queries/main.py          # 直接运行
"""

from .runner import run_tests

if __name__ == "__main__":
    results = run_tests()
    total = len(results)
    ok = sum(1 for r in results if r["error"] is None)
    print(f"\n完成: {ok}/{total} 条用例通过")
