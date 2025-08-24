# utils/station_name_normalizer.py (最終修正版)
import re

def normalize_station_name(name: str) -> str:
    """
    一個純粹的站名標準化工具 v2.1。
    新增了對「台北車」的自動修正和對「台北車站」的特殊處理。
    """
    if not isinstance(name, str):
        return ""
    
    normalized_input = name.lower().strip().replace("臺", "台")

    # ✨ 核心修正一：如果輸入是「台北車」，直接修正為「台北車站」
    if normalized_input == "台北車":
        return "台北車站"
        
    # ✨ 核心修正二：如果名稱包含 "台北車站"，則直接回傳標準化的全名
    if "台北車站" in normalized_input:
        return "台北車站"
    
    # 對於其他站名，才執行後續的標準化流程
    normalized_input = re.sub(r"[\(（].*?[\)）]", "", normalized_input).strip()
    
    if normalized_input.endswith("站"):
        normalized_input = normalized_input.removesuffix("站")
        
    return normalized_input