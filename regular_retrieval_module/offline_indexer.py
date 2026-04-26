"""
常规检索模块的离线索引器
使用共享CLIP编码器预建图像特征向量库（支持磁盘缓存）
"""

import os
import time
from typing import List, Dict, Any
from PIL import Image
import requests
import torch

from shared.clip_encoder import CLIPEncoder
from .constants import CACHE_DIR


class OfflineIndexer:
    """离线索引器，用于预建图像特征向量库（支持磁盘缓存加速冷启动）"""

    def __init__(self, clip_encoder: CLIPEncoder):
        self.clip_encoder = clip_encoder
        self.image_feature_db: Dict[str, Dict[str, Any]] = {}
        self._cache_path = os.path.join(CACHE_DIR, "image_features.pt")

    def _load_cache(self, image_urls: List[str]) -> bool:
        """尝试从磁盘缓存加载向量库。URL 列表完全一致才命中。"""
        if not os.path.exists(self._cache_path):
            return False
        try:
            data = torch.load(self._cache_path, map_location="cpu", weights_only=False)
            cached_urls = set(data.get("urls", []))
            if cached_urls == set(image_urls):
                self.image_feature_db = data.get("features", {})
                print(f"[常规检索] 从磁盘缓存加载 {len(self.image_feature_db)} 条向量")
                return True
            else:
                print(f"[常规检索] 缓存 URL 列表不匹配，重新索引")
                return False
        except Exception as e:
            print(f"[常规检索] 缓存加载失败: {e}，重新索引")
            return False

    def _save_cache(self, image_urls: List[str]):
        """将向量库保存到磁盘（仅保存特征向量，不保存原图）"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        slim_db = {url: {"image_feature": data["image_feature"]}
                   for url, data in self.image_feature_db.items()}
        torch.save({"urls": image_urls, "features": slim_db}, self._cache_path)
        size_mb = os.path.getsize(self._cache_path) / 1024 / 1024
        print(f"[常规检索] 向量缓存已保存: {self._cache_path} ({size_mb:.1f} MB)")

    def index(self, image_urls: List[str], batch_size: int = 128) -> float:
        """
        离线阶段：预建图像特征向量库
        流程：图像库 → CLIP图像编码器 → 离线向量库
        分批加载+编码+存储，实时刷新进度条覆盖完整耗时。

        Returns:
            总用时（秒）
        """
        total = len(image_urls)
        print(f"[常规检索] 开始离线索引 {total} 张图像...")

        # 尝试从磁盘缓存加载，命中则跳过重编码
        if self._load_cache(image_urls):
            return 0.0

        bar_width = 30
        start_time = time.time()
        valid_count = 0

        for batch_start in range(0, total, batch_size):
            batch_urls = image_urls[batch_start:batch_start + batch_size]

            # 1. 加载当前批次的图像
            valid_images = []
            valid_urls = []
            for url in batch_urls:
                try:
                    if url.startswith("http"):
                        image_raw = Image.open(requests.get(url, stream=True, timeout=10).raw)
                    else:
                        image_raw = Image.open(url)
                    valid_images.append(image_raw)
                    valid_urls.append(url)
                except Exception as e:
                    print(f"\n  [ERROR] [{url}] 加载失败: {str(e)}")

            # 2. 编码并存入向量库
            if valid_images:
                batch_features = self.clip_encoder.encode_images(valid_images)
                if batch_features is not None:
                    for i, url in enumerate(valid_urls):
                        self.image_feature_db[url] = {
                            "image_feature": batch_features[i].cpu(),
                            "raw_image": valid_images[i]
                        }
                        valid_count += 1

            # 3. 刷新进度条（覆盖加载+编码+存储的完整耗时）
            processed = min(batch_start + batch_size, total)
            elapsed = time.time() - start_time
            pct = processed / total
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            eta = (elapsed / pct - elapsed) if pct > 0 else 0
            print(
                f"\r  [进度] |{bar}| {processed}/{total} ({pct:.0%}) "
                f"耗时 {elapsed:.1f}s 预计剩余 {eta:.1f}s",
                end="", flush=True
            )

        total_time = time.time() - start_time
        print()
        print(f"[常规检索] 离线索引完成，成功 {valid_count} 张，总用时 {total_time:.1f}s")
        self._save_cache(image_urls)
        return total_time

    def get_db(self) -> Dict[str, Dict[str, Any]]:
        """获取离线向量库"""
        return self.image_feature_db

    def is_empty(self) -> bool:
        """检查向量库是否为空"""
        return len(self.image_feature_db) == 0