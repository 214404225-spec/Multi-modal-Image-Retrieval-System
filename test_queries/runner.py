"""
测试运行器
执行测试用例并保存结果
"""

import sys
import os
import time

from .test_data import TEST_QUERIES


def run_tests():
    """运行全部测试查询，返回结果列表"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from agent_pipeline import MultiModalAgentPipeline

    pipeline = MultiModalAgentPipeline()

    results = []
    total = len(TEST_QUERIES)

    print("=" * 60)
    print(f"多模态图像检索系统 - 自动化测试 ({total} 条用例)")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{total}] 查询: {query}")
        print("-" * 40)

        start = time.time()
        try:
            result = pipeline.chat(query)
            elapsed = time.time() - start
            results.append({
                "id": i,
                "input": query,
                "output": result.get("output", ""),
                "elapsed": elapsed,
                "error": None,
            })
            print(f"完成 ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "id": i,
                "input": query,
                "output": "",
                "elapsed": elapsed,
                "error": str(e),
            })
            print(f"错误: {e}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    success_count = sum(1 for r in results if r["error"] is None)
    print(f"通过: {success_count}/{total}")

    for r in results:
        status = "OK" if r["error"] is None else f"ERR: {r['error']}"
        print(f"\n[{r['id']:2d}] {status} ({r['elapsed']:.1f}s)")
        print(f"      输入: {r['input']}")
        output_preview = r["output"][:200] if r["output"] else "(无输出)"
        print(f"      输出: {output_preview}")

    # 保存到文件
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("多模态图像检索系统 - 自动化测试结果\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"通过: {success_count}/{total}\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"[{r['id']:2d}] 输入: {r['input']}\n")
            if r["error"]:
                f.write(f"      错误: {r['error']}\n")
            else:
                f.write(f"      输出: {r['output']}\n")
            f.write(f"      耗时: {r['elapsed']:.1f}s\n")
            f.write("-" * 60 + "\n")

    print(f"\n结果已保存到 test_results.txt")
    return results
