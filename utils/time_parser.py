import re
import logging

logger = logging.getLogger(__name__)

def parse_countdown_to_seconds(countdown_str: str) -> float:
    """
    將倒數時間字串轉換為秒數。
    對於非預期格式或錯誤，返回浮點無限大，以便在排序時將其排在最後。
    
    Args:
        countdown_str (str): 列車倒數計時的字串，例如 "5分30秒", "進站", "已到站", "已離站"。

    Returns:
        float: 轉換後的秒數。如果無法解析或表示列車已離站/無效，則返回 float('inf')。
    """
    if not isinstance(countdown_str, str):
        return float('inf')

    # 優先處理特殊狀態
    if '進站' in countdown_str or '已到站' in countdown_str:
        return 0.0 # 列車進站或已到站，給予最高優先級
    if '已離站' in countdown_str:
        return float('inf') # 列車已離站，給予最低優先級

    # 嘗試解析 "X 分 Y 秒" 格式
    m = re.search(r'(?:(\d+)\s*分)?\s*(?:(\d+)\s*秒)?', countdown_str)
    if m and (m.group(1) or m.group(2)):
        minutes = int(m.group(1) or 0)
        seconds = int(m.group(2) or 0)
        return float(minutes * 60 + seconds)
    
    # 如果是純數字字串，嘗試直接轉換
    try:
        return float(countdown_str)
    except (ValueError, TypeError):
        return float('inf') # 無法解析則排在最後
