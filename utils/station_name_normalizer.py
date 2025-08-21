# utils/station_name_normalizer.py
"""
======================================================================
|           MetroPet AI Agent - Station Name Normalizer            |
|                   (採用 finyster 分支的純函式架構)                 |
======================================================================
此檔案提供一個純粹的站點名稱標準化工具函式。
它的唯一職責是接收一個字串，並對其進行一系列的清理和格式化，
以便後續的服務 (如 StationManager) 能夠進行精確的資料庫比對。
"""

import re

def normalize_station_name(name: str) -> str:
    """
    一個純粹的站點名稱標準化工具。
    功能：轉小寫、移除頭尾空白、移除括號內容、移除'站'字尾、繁轉簡。
    它只負責處理字串，不進行任何查核。
    """
    if not isinstance(name, str):
        return ""
    
    # 轉小寫、移除頭尾空白、繁轉簡
    normalized_input = name.lower().strip().replace("臺", "台")
    
    # 移除括號及其內容
    normalized_input = re.sub(r"[\(（].*?[\)）]", "", normalized_input).strip()
    
    # 【✨核心修正✨】使用更安全的方式移除字尾 "站"
    if normalized_input.endswith("站"):
        normalized_input = normalized_input.removesuffix("站")
        
    return normalized_input
