"""
Agent Pipeline主入口
提供交互式对话功能
"""

from .pipeline import MultiModalAgentPipeline


def main():
    """主函数，启动交互式对话"""
    print("=" * 50)
    print("多模态图像检索系统")
    print("=" * 50)
    
    # 初始化 Pipeline（模型由 INTENT_MODEL / VL_MODEL 环境变量控制）
    pipeline = MultiModalAgentPipeline()
    
    # 交互式对话
    print("\n输入 'quit' 或 'exit' 退出")
    print("提示：检索不在常见类别中的物体时，不加属性条件直接走常规检索更高效。")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("再见！")
            break
        if user_input:
            result = pipeline.chat(user_input)
            print(f"\nAgent: {result['output']}")


if __name__ == "__main__":
    main()