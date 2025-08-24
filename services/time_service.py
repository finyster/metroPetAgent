# services/time_service.py
import logging
from datetime import datetime, timedelta
from typing import Optional
import dateparser

logger = logging.getLogger(__name__)

class TimeService:
    """
    一個專門處理各種自然語言時間字串解析的服務。
    集中管理所有時間相關的複雜邏輯。
    """
    def __init__(self):
        logger.info("--- [TimeService] 正在初始化時間解析服務... ---")

    def parse_datetime(self, datetime_str: Optional[str] = None) -> datetime:
        """
        【v1.1 核心解析器 - 整點強化版】
        將各種格式的時間字串（包含中文、相對時間）解析為標準的 datetime 物件。
        如果解析失敗或未提供字串，則回傳當前時間。

        Args:
            datetime_str (Optional[str]): 使用者輸入的時間字串。

        Returns:
            datetime: 解析後的標準時間物件。
        """
        now = datetime.now()

        if not datetime_str:
            return now

        if datetime_str.lower() in ["現在", "即將", "馬上", "下一班車"]:
            return now

        # 1. 標準化中文單位
        processed_str = datetime_str.replace('年', '-').replace('月', '-').replace('日', ' ')
        processed_str = processed_str.replace('點', ':').replace('分', '')

        # --- ✨✨✨【核心修正點】✨✨✨
        # 處理像 "17點" 這樣轉換後變成 "17:" 的不完整格式
        # 我們手動在後面補上 "00"，使其變為 "17:00"
        if processed_str.strip().endswith(':'):
            processed_str = processed_str.strip() + '00'
        # --- ✨✨✨【修正結束】✨✨✨

        # 2. 處理關鍵相對詞彙
        base_date = now
        time_part = processed_str

        if '明天' in processed_str:
            base_date = now + timedelta(days=1)
            time_part = processed_str.replace('明天', '').strip()
        elif '今天' in processed_str:
            time_part = processed_str.replace('今天', '').strip()
        elif '後天' in processed_str:
            base_date = now + timedelta(days=2)
            time_part = processed_str.replace('後天', '').strip()

        # 3. 使用 dateparser 進行最終解析
        try:
            parsed_dt = dateparser.parse(
                time_part,
                languages=['zh'],
                settings={
                    'RELATIVE_BASE': base_date,
                    'TIMEZONE': 'Asia/Taipei'
                }
            )
            if parsed_dt:
                logger.info(f"--- [TimeService] 成功將 '{datetime_str}' 解析為 {parsed_dt.strftime('%Y-%m-%d %H:%M:%S')} ---")
                return parsed_dt
        except Exception as e:
            logger.error(f"--- [TimeService] 解析 '{datetime_str}' 時發生錯誤: {e} ---")

        logger.warning(f"--- [TimeService] 無法解析 '{datetime_str}'，將使用當前時間作為替代。 ---")
        return now

# 建立 TimeService 的一個全域實例
time_service = TimeService()