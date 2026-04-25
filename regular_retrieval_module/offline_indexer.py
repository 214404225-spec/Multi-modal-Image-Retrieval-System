"""
常规检索模块的离线索引器
负责预建图像特征向量库
"""

from typing import List, Dict, Any
from PIL import Image
import requests

from .image_encoder import ClipViTImageEncoder


class OfflineIndexer:
    """离线索引器，用于预建图像特征向量库"""
    
    def __init__(self, image_encoder: ClipViTImageEncoder):
        self.image_encoder = image_encoder
        self.image_feature_db: Dict[str, Dict[str, Any]] = {}
    
    def index(self, image_urls: List[str], batch_size: int = None) -> None:
        """
        离线阶段：预建图像特征向量库（支持批处理加速）
        流程：图像库 → CLIP图像编码器 → 离线向量库1
        
        Args:
            image_urls: 图像URL或路径列表
            batch_size: 批处理大小（默认自动检测：16G显存=128，8G=64，CPU=16）
        """
        # 自动检测显存并设置batch size
        if batch_size is None:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory_gb >= 16:
                    batch_size = 128
                elif gpu_memory_gb >= 8:
                    batch_size = 64
                else:
                    batch_size = 32
                print(f"[常规检索] 检测到{gpu_memory_gb:.1f}GB显存，自动设置batch_size={batch_size}")
            else:
                batch_size = 16
                print(f"[常规检索] 未检测到GPU，使用CPU模式，batch_size={batch_size}")
        
        print(f"[常规检索] 开始离线索引 {len(image_urls)} 张图像（批处理大小={batch_size}）...")
        
        # 收集所有有效的图像
        valid_images = []
        valid_urls = []
        
        for url in image_urls:
            try:
                # 获取图像（只加载一次）
                if url.startswith("http"):
                    image_raw = Image.open(requests.get(url, stream=True, timeout=10).raw)
                else:
                    image_raw = Image.open(url)
                valid_images.append(image_raw)
                valid_urls.append(url)
            except Exception as e:
                print(f"  [ERROR] [{url}] 加载失败: {str(e)}")
        
        # 批处理编码
        total = len(valid_images)
        for i in range(0, total, batch_size):
            batch_images = valid_images[i:i+batch_size]
            batch_urls = valid_urls[i:i+batch_size]
            
            # 批量编码
            batch_features = self.image_encoder.encode_images(batch_images)
            
            if batch_features is not None:
                for j, url in enumerate(batch_urls):
                    self.image_feature_db[url] = {
                        "image_feature": batch_features[j].cpu(),
                        "raw_image": batch_images[j]
                    }
                    print(f"  [OK] [{url}] 索引完成 ({i+j+1}/{total})")
            else:
                for url in batch_urls:
                    print(f"  [FAIL] [{url}] 编码失败")
        
        print(f"[常规检索] 离线索引完成，成功 {len(self.image_feature_db)} 张")
    
    def get_db(self) -> Dict[str, Dict[str, Any]]:
        """获取离线向量库"""
        return self.image_feature_db
    
    def is_empty(self) -> bool:
        """检查向量库是否为空"""
        return len(self.image_feature_db) == 0