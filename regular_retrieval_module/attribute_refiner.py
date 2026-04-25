"""
常规检索模块的属性精排序器
根据属性条件对粗检索结果进行重排序
"""

from typing import List, Dict, Optional

from .text_encoder import ChineseRobertaTextEncoder
from .image_encoder import ClipViTImageEncoder


class AttributeRefiner:
    """属性精排序器，用于根据属性条件重排序检索结果"""
    
    def __init__(self, text_encoder: ChineseRobertaTextEncoder, image_encoder: ClipViTImageEncoder):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
    
    def refine(self, results: List[Dict], category: str, 
               attributes: List[str], top_k: int = None,
               alpha: float = 0.4, beta: float = 0.6) -> List[Dict]:
        """
        两阶段检索的第二阶段：根据属性条件重排序
        使用Taiyi-CLIP编码"类别+属性"进行细粒度匹配
        
        Args:
            results: 粗检索结果列表
            category: 类别名称
            attributes: 属性条件列表（如["棕色", "站立", "清晰"]）
            top_k: 返回的结果数量
            alpha: 粗检索分数权重（默认0.4）
            beta: 属性分数权重（默认0.6）
            
        Returns:
            按属性条件重排序后的结果列表
        """
        if not attributes or not results:
            return results
        
        # 构建属性查询文本，使用Taiyi-CLIP编码
        attr_query = f"{category}的{'、'.join(attributes)}"
        query_feature = self.text_encoder.encode(attr_query)
        
        logit_scale = self.image_encoder.get_logit_scale()
        
        # 重新计算相似度并融合分数
        for r in results:
            img_feat = self.image_encoder.encode(r["url"])
            if img_feat is not None:
                attr_score = (logit_scale * img_feat @ query_feature.t()).item()
                r["attr_score"] = attr_score
                
                # 加权融合：final_score = alpha * category_score + beta * attr_score
                category_score = r.get("score", 0)
                r["final_score"] = alpha * category_score + beta * attr_score
            else:
                # 如果图像编码失败，保留原始分数
                r["attr_score"] = r.get("score", 0)
                r["final_score"] = r.get("score", 0)
        
        # 按融合分数重排序
        refined = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)
        return refined[:top_k] if top_k else refined
