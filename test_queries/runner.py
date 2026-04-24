"""
测试运行器
执行测试用例并保存结果
"""

import sys
import os

from .test_data import TEST_QUERIES


def run_tests():
    """运行测试查询"""
    # 添加父目录到路径以便导入agent_pipeline
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from agent_pipeline import MultiModalAgentPipeline
    
    # 初始化Pipeline（使用Ollama）
    pipeline = MultiModalAgentPipeline(model_name="qwen2.5:3b")
    
    results = []
    
    print("=" * 80)
    print("多模态图像检索系统 - 测试用例 (Ollama qwen2.5:3b)")
    print("=" * 80)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/25")
        print(f"{'='*60}")
        
        result = pipeline.chat(query)
        
        results.append({
            "id": i,
            "input": query,
            "output": result["output"]
        })
        
        print(f"\n输入: {query}")
        print(f"输出: {result['output']}")
        print("-" * 60)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("测试结果汇总 - 25个输入-输出对")
    print("=" * 80)
    
    for r in results:
        print(f"\n[{r['id']:2d}] 输入: {r['input']}")
        print(f"      输出: {r['output']}")
    
    # 保存到文件
    print("\n" + "=" * 80)
    print("保存结果到文件...")
    print("=" * 80)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("多模态图像检索系统 - 25个测试用例结果 (Ollama qwen2.5:3b)\n")
        f.write("=" * 60 + "\n\n")
        
        for r in results:
            f.write(f"[{r['id']:2d}] 输入: {r['input']}\n")
            f.write(f"      输出: {r['output']}\n")
            f.write("-" * 60 + "\n")
    
    print("结果已保存到 test_results.txt")
    
    return results