"""
细粒度检索模块的离线索引器
使用VL模型识别图像内容，然后用CLIP编码描述文本
"""

from typing import List, Dict, Any

from .vl_models import VLModelManager
from .clip_encoder import CLIPEncoder


class OfflineIndexer:
    """细粒度离线索引器"""
    
    def __init__(self, vl_manager: VLModelManager, clip_encoder: CLIPEncoder):
        self.vl_manager = vl_manager
        self.clip_encoder = clip_encoder
        self.image_feature_db: Dict[str, Dict[str, Any]] = {}
    
    def index(self, image_paths: List[str]) -> None:
        """
        离线阶段：处理离线图片库。
        流程：图片库 → Qwen2.5-VL (生成标签和数量) → CLIP文本编码器 → 离线向量库2
        """
        print(f"[细粒度检索] 开始离线索引 {len(image_paths)} 张图像...")
        for path in image_paths:
            try:
                # 1. 使用VL模型抽取细粒度特征
                labels_counts = self.vl_manager.extract_features(path)
                
                if labels_counts:
                    # 2. 将字典拼成一句话，交给 CLIP 提取特征
                    description = "，".join([f"{count}个{label}" for label, count in labels_counts.items()])
                    
                    # 3. CLIP文本编码提取向量
                    text_feature = self.clip_encoder.encode_text(description)
                    
                    # 4. 存入向量库
                    self.image_feature_db[path] = {
                        "description": description,
                        "labels_counts": labels_counts,
                        "vector": text_feature.cpu()
                    }
                    print(f"  [OK] [{path}] 离线特征索引完成: {description}")
                else:
                    print(f"  [SKIP] [{path}] 未识别到物体")
                    
            except Exception as e:
                print(f"  [ERROR] [{path}] 索引失败: {str(e)}")
        
        print(f"[细粒度检索] 离线索引完成，成功 {len(self.image_feature_db)} 张")
    
    def get_db(self) -> Dict[str, Dict[str, Any]]:
        """获取离线向量库"""
        return self.image_feature_db
    
    def is_empty(self) -> bool:
        """检查向量库是否为空"""
        return len(self.image_feature_db) == 0