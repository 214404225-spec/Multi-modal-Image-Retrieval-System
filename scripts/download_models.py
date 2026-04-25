"""
模型下载脚本
用于从国内镜像源下载多模态图像检索系统所需的预训练模型到本地
下载完成后，系统启动将不再需要联网加载模型

使用方法:
    python scripts/download_models.py                          # 使用默认镜像源下载
    python scripts/download_models.py --output-dir ./models    # 指定下载目录
    python scripts/download_models.py --mirror official        # 使用官方源
    python scripts/download_models.py --mirror hf-mirror       # 使用hf-mirror镜像
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


# 模型配置
MODELS = {
    "clip_ViT": {
        "model_id": "openai/clip-vit-large-patch14",
        "description": "CLIP ViT-L/14 图像编码器（用于图像特征提取）",
    },
    "Chinese_RoBERTa": {
        "model_id": "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese",
        "description": "Taiyi-CLIP RoBERTa 文本编码器（用于中文文本特征提取）",
    },
}

# 镜像源配置
MIRROR_SOURCES = {
    "hf-mirror": "https://hf-mirror.com",
    "official": "https://huggingface.co",
}


def download_model(model_key: str, output_dir: str, mirror: str = "hf-mirror"):
    """
    下载单个模型到本地
    
    Args:
        model_key: 模型键名（如 'clip_ViT'）
        output_dir: 输出目录
        mirror: 镜像源名称
    """
    model_config = MODELS[model_key]
    model_id = model_config["model_id"]
    description = model_config["description"]
    
    target_dir = os.path.join(output_dir, model_key)
    
    print(f"\n{'='*60}")
    print(f"下载模型: {model_key}")
    print(f"模型ID: {model_id}")
    print(f"描述: {description}")
    print(f"目标路径: {target_dir}")
    print(f"镜像源: {mirror} ({MIRROR_SOURCES.get(mirror, 'unknown')})")
    print(f"{'='*60}")
    
    # 检查是否已存在
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "config.json")):
        print(f"[INFO] 模型 {model_key} 已存在于本地: {target_dir}")
        print(f"[INFO] 跳过下载（如需重新下载，请删除该目录）")
        return True
    
    try:
        from huggingface_hub import snapshot_download
        
        # 设置镜像源
        if mirror == "hf-mirror":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        print(f"[INFO] 开始下载模型 {model_id}...")
        print(f"[INFO] 这可能需要几分钟时间，请耐心等待...")
        
        # 下载模型
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
        )
        
        print(f"[SUCCESS] 模型 {model_key} 下载成功!")
        print(f"[INFO] 保存路径: {target_dir}")
        return True
        
    except ImportError:
        print("[ERROR] 未安装 huggingface_hub 库")
        print("[INFO] 请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] 模型 {model_key} 下载失败: {str(e)}")
        return False


def verify_model(model_key: str, output_dir: str) -> bool:
    """
    验证模型文件是否完整
    
    Args:
        model_key: 模型键名
        output_dir: 输出目录
        
    Returns:
        是否验证通过
    """
    target_dir = os.path.join(output_dir, model_key)
    
    if not os.path.exists(target_dir):
        print(f"[WARN] 模型目录不存在: {target_dir}")
        return False
    
    # 必需文件列表
    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(target_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"[WARN] 模型 {model_key} 缺少文件: {', '.join(missing_files)}")
        return False
    
    print(f"[OK] 模型 {model_key} 验证通过")
    return True


def update_constants(output_dir: str):
    """
    更新 constants.py 中的本地模型路径配置
    
    Args:
        output_dir: 模型输出目录
    """
    constants_path = os.path.join(os.path.dirname(output_dir), "regular_retrieval_module", "constants.py")
    
    # 如果 constants.py 不存在，尝试在项目根目录查找
    if not os.path.exists(constants_path):
        constants_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "regular_retrieval_module", "constants.py")
    
    if not os.path.exists(constants_path):
        print(f"[WARN] 未找到 constants.py 文件，请手动更新配置")
        return
    
    # 读取原文件
    with open(constants_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 更新 LOCAL_MODEL_PATHS
    new_paths = {
        "Chinese_RoBERTa": os.path.join(output_dir, "Chinese_RoBERTa").replace("\\", "/"),
        "clip_ViT": os.path.join(output_dir, "clip_ViT").replace("\\", "/"),
    }
    
    # 构建新的 LOCAL_MODEL_CACHE 配置
    cache_config = "{\n"
    for key, path in new_paths.items():
        cache_config += f"    '{key}': '{path}',\n"
    cache_config += "}"
    
    # 替换原有配置
    import re
    pattern = r"LOCAL_MODEL_CACHE\s*=\s*\{[^}]*\}"
    replacement = f"LOCAL_MODEL_CACHE = {cache_config}"
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(constants_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"[INFO] 已更新 {constants_path} 中的 LOCAL_MODEL_CACHE 配置")
    else:
        print(f"[WARN] 未能自动更新 constants.py，请手动更新配置")


def main():
    parser = argparse.ArgumentParser(
        description="下载多模态图像检索系统所需的预训练模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_models.py                          # 使用默认配置下载
  python scripts/download_models.py --output-dir ./models    # 指定下载目录
  python scripts/download_models.py --mirror official        # 使用官方源
  python scripts/download_models.py --verify-only            # 仅验证已下载的模型
        """,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="模型下载目录 (默认: ./models)",
    )
    
    parser.add_argument(
        "--mirror",
        type=str,
        choices=["hf-mirror", "official"],
        default="hf-mirror",
        help="HuggingFace镜像源 (默认: hf-mirror)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default=None,
        help="指定下载单个模型 (默认: 下载所有模型)",
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证已下载的模型，不执行下载",
    )
    
    parser.add_argument(
        "--update-constants",
        action="store_true",
        help="下载完成后自动更新 constants.py 配置",
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("多模态图像检索系统 - 模型下载工具")
    print("=" * 60)
    print(f"下载目录: {os.path.abspath(args.output_dir)}")
    print(f"镜像源: {args.mirror} ({MIRROR_SOURCES.get(args.mirror, 'unknown')})")
    
    # 确定要下载的模型列表
    models_to_process = [args.model] if args.model else list(MODELS.keys())
    
    success_count = 0
    fail_count = 0
    
    for model_key in models_to_process:
        if args.verify_only:
            if verify_model(model_key, args.output_dir):
                success_count += 1
            else:
                fail_count += 1
        else:
            if download_model(model_key, args.output_dir, args.mirror):
                success_count += 1
            else:
                fail_count += 1
    
    # 验证所有模型
    print(f"\n{'='*60}")
    print("模型验证结果:")
    print(f"{'='*60}")
    
    for model_key in models_to_process:
        verify_model(model_key, args.output_dir)
    
    # 更新 constants.py
    if args.update_constants and success_count == len(models_to_process):
        print(f"\n{'='*60}")
        print("更新配置文件...")
        update_constants(args.output_dir)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("下载完成!")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"{'='*60}")
    
    if fail_count == 0:
        print("\n后续操作:")
        print(f"1. 确保 constants.py 中的 LOCAL_MODEL_CACHE 指向: {os.path.abspath(args.output_dir)}")
        print("2. 运行系统: python -m agent_pipeline.main")
        print("3. 系统将使用本地模型，无需联网")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())