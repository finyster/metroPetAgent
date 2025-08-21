# services/station_service.py
"""
======================================================================
|               MetroPet AI Agent - Station Manager                |
|                 (融合 finyster & alan 分支功能)                  |
======================================================================
此服務是 AI 助理辨識捷運站名的核心大腦。
它融合了兩大分支的優點：
- alan/main: 提供完整的方向解析、終點站查詢等核心業務邏輯。
- finyster: 提供了強大的「向量搜尋」能力，讓 AI 能理解模糊或錯誤的站名。
"""

# ---------------------------------------------------------------------
# 1. 核心模組匯入 (Core Module Imports)
# ---------------------------------------------------------------------
import json
import logging
import os
import re
from typing import Any, Dict, List, Union

import config
from utils.exceptions import StationNotFoundError
from utils.station_name_normalizer import normalize_station_name

# ---------------------------------------------------------------------
# 2. 基本設定 (Basic Configuration)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 3. StationManager 類別定義
# ---------------------------------------------------------------------
class StationManager:
    """
    管理捷運站點資料，包括名稱、ID、別名和方向的解析。
    """
    def __init__(self, station_data_path: str):
        self.station_data_path = station_data_path
        self.station_map = self._load_station_data()
        # official_name_map 用於從標準化名稱反查官方名稱，對於提供友善提示很重要
        self.official_name_map: Dict[str, str] = self._build_official_name_map()

    def _load_station_data(self) -> dict:
        """
        從本地檔案載入由 build_database.py 預先建立的站點資料。
        """
        if os.path.exists(self.station_data_path) and os.path.getsize(self.station_data_path) > 0:
            try:
                with open(self.station_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data:
                    logger.info(f"--- ✅ 已從 {os.path.basename(self.station_data_path)} 載入站點資料 ---")
                    return data
            except Exception as e:
                logger.error(f"--- ❌ 讀取站點資料失敗 ({e}) ---")
        
        logger.warning(f"--- ⚠️ 本地站點資料 '{self.station_data_path}' 不存在或為空，建議執行 build_database.py 來生成。 ---")
        return {}
    
    def _build_official_name_map(self) -> Dict[str, str]:
        """
        [alan 的關鍵邏輯] 建立一個從標準化名稱到官方原始名稱的反向映射表。
        """
        if not self.station_map: return {}
        
        id_to_name = {}
        # 優先使用純中文的名稱作為官方名稱
        for name, ids in self.station_map.items():
            if not re.search('[a-zA-Z0-9]', name):
                for station_id in ids:
                    if station_id not in id_to_name:
                        id_to_name[station_id] = name.replace("台", "臺") # 轉回官方用字
        
        official_map = {}
        for name, ids in self.station_map.items():
             if ids and ids[0] in id_to_name:
                 official_map[name] = id_to_name[ids[0]]
        
        logger.info(f"--- ✅ 已建立 {len(official_map)} 筆標準化名稱 -> 官方名稱的映射。 ---")
        return official_map

    def get_station_ids(self, station_name: str) -> Union[List[str], Dict[str, Any], None]:
        """
        【finyster 的核心智慧搜尋邏輯】
        1. 優先精準比對（已包含 build_database.py 建立的豐富別名）。
        2. 若失敗，啟用向量語意搜尋作為備用方案。
        3. 根據相似度分數，決定是直接採用還是向使用者提出建議。
        """
        if not station_name: return None
        norm_name = normalize_station_name(station_name)
        if not norm_name: return None

        # 步驟 1: 精準比對
        if norm_name in self.station_map:
            return self.station_map[norm_name]

        # 步驟 2: 向量語意搜尋 (Fallback)
        from services import service_registry # 延遲匯入以避免循環依賴
        logger.warning(f"--- 精準比對「{norm_name}」失敗，啟用向量語意搜尋... ---")
        vector_service = service_registry.vector_search_service
        if not vector_service or not vector_service.is_ready:
             logger.error("--- ❌ 向量搜尋服務未就緒，無法進行模糊查詢。 ---")
             return None

        best_match_info = vector_service.find_most_similar(norm_name)
        if best_match_info:
            match_name, score = best_match_info
            logger.info(f"--- 向量搜尋結果: 最相似「{match_name}」，分數: {score:.4f} ---")
            if score >= 0.7:
                logger.info(f"--- 分數 > 0.7，直接採用「{match_name}」。 ---")
                return self.station_map.get(match_name)
            elif score >= 0.4:
                logger.info(f"--- 分數 0.4-0.7，將「{match_name}」作為建議返回。 ---")
                return {"suggestion": match_name, "original_query": station_name}
        
        logger.error(f"--- ❌ 向量搜尋分數過低或無匹配: '{norm_name}' ---")
        return None

    def get_official_unnormalized_name(self, normalized_name: str) -> str:
        """
        [alan 的關鍵邏輯] 根據標準化後的站名，回傳其原始的官方全名。
        """
        return self.official_name_map.get(normalized_name, normalized_name)

    def get_all_station_names(self) -> List[str]:
        """
        回傳所有不含英文和別名的官方中文站名列表，用於盤點。
        """
        # 從 station_map 的鍵中篩選出純中文的站名
        return sorted([name for name in self.station_map.keys() if not re.search('[a-zA-Z0-9]', name)])


# ---------------------------------------------------------------------
# 4. 單一實例 (Singleton Instance)
# ---------------------------------------------------------------------
# 建立 StationManager 的一個全域實例，供 ServiceRegistry 和其他服務使用。
station_manager = StationManager(config.STATION_DATA_PATH)