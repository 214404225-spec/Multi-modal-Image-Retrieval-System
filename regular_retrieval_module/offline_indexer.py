"""
常规检索模块的离线索引器
负责预建图像特征向量库
"""

from typing import List, Dict, Any
from PIL import Image
import requests

from .image_encoder import CLIPImageEncoder


class OfflineIndexer:
    """离线索引器，用于预建图像特征向量库"""
    
    def __init__(self, image_encoder: CLIPImageEncoder):
        self.image_encoder = image_encoder
        self.image_feature_db: Dict[str, Dict[str, Any]] = {}
    
    def index(self, image_urls: List[str]) -> None:
        """
        离线阶段：预建图像特征向量库
        流程：图像库 → CLIP图像编码器 → 离线向量库1
        
        Args:
            image_urls: 图像URL或路径列表
        """
        print(f"[常规检索] 开始离线索引 {len(image_urls)} 张图像...")
        for url in image_urls:
            try:
                # 获取图像
                if url.startswith("http"):
                    image_raw = Image.open(requests.get(url, stream=True, timeout=10).raw)
                else:
                    image_raw = Image.open(url)
                
                # 使用CLIP图像编码器提取特征
                image_features = self.image_encoder.encode(url)
                
                if image_features is not None:
                    # 存入离线向量库
                    self.image_feature_db[url] = {
                        "image_feature": image_features.cpu(),
                        "raw_image": image_raw
                    }
                    print(f"  [OK] [{url}] 索引完成")
                else:
                    print(f"  [FAIL] [{url}] 编码失败")
                    
            except Exception as e:
                print(f"  [ERROR] [{url}] 索引失败: {str(e)}")
        
        print(f"[常规检索] 离线索引完成，成功 {len(self.image_feature_db)} 张")
    
    def get_db(self) -> Dict[str, Dict[str, Any]]:
        """获取离线向量库"""
        return self.image_feature_db
    
    def is_empty(self) -> bool:
        """检查向量库是否为空"""
        return len(self.image_feature_db) == 0