# services/llm_time_parser_service.py
import logging
from datetime import datetime
from langchain_community.chat_models import ChatOllama
import dateparser # 導入 dateparser

logger = logging.getLogger(__name__)

class LLMTimeParserService:
    """
    一個使用本地 Ollama LLM 來解析自然語言時間的服務。
    採用 LLM (理解) + Parser (解析) 的兩階段策略，以達到最佳穩定性。
    """
    def __init__(self, model_name: str = "llama3:8b"):
        self.model = None
        logger.info(f"--- [OllamaTimeParser] 正在初始化 Ollama 時間解析服務 (使用模型: {model_name})... ---")
        try:
            self.model = ChatOllama(model=model_name, temperature=0.0)
            logger.info(f"--- ✅ [OllamaTimeParser] Ollama 模型 '{model_name}' 已設定完成。 ---")
            logger.info("--- 請確保您已在另一個終端機視窗執行 `ollama run {model_name}` 來啟動模型。 ---")
        except Exception as e:
            logger.error(f"--- ❌ [OllamaTimeParser] 初始化 Ollama 模型 '{model_name}' 失敗: {e} ---")

    def parse_datetime(self, datetime_str: str = None) -> datetime:
        """
        使用本地 LLM 將口語化時間轉換為標準化字串，再由 dateparser 解析。
        """
        now = datetime.now()
        
        if not datetime_str or datetime_str.lower() in ["現在", "即將", "馬上", "下一班車"]:
            return now
            
        if not self.model:
            logger.error("--- ❌ [OllamaTimeParser] 模型未初始化，將回傳當前時間。 ---")
            return now

        # ✨ 1. 設計一個新的 Prompt，讓 LLM 進行「語言轉換」而非「格式轉換」
        prompt = f"""
        You are an expert multilingual time normalization assistant. Your task is to convert a user's time description into a simple, standardized English format.

        # Instructions
        - Analyze the "User's Time Description" in the context of the "Current Time".
        - Convert the description into a simple machine-readable format like "YYYY-MM-DD HH:MM", "tomorrow HH:MM", "today HH:MM", "in 2 hours", etc.
        - Handle Chinese terms like '明天' (tomorrow), '早上' (morning/am), '下午' (afternoon/pm), '晚上' (evening/pm).
        - Your response MUST ONLY contain the simplified time string. DO NOT add any explanations.

        # Reference Information
        - Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
        - User's Time Description: "{datetime_str}"

        # Your Simplified Output:
        """

        try:
            # ✨ 2. 讓 LLM 輸出一個更容易處理的標準化字串
            response = self.model.invoke(prompt)
            simplified_time_str = response.content.strip()
            
            logger.info(f"--- [OllamaTimeParser] LLM 標準化結果: '{simplified_time_str}' ---")

            # ✨ 3. 將標準化後的字串，交給專業的 dateparser 進行最終解析
            parsed_dt = dateparser.parse(
                simplified_time_str,
                settings={'PREFER_DATES_FROM': 'future', 'TIMEZONE': 'Asia/Taipei'}
            )

            if parsed_dt:
                logger.info(f"--- [OllamaTimeParser] 成功將 '{datetime_str}' 解析為 {parsed_dt.strftime('%Y-%m-%d %H:%M:%S')} ---")
                return parsed_dt
            else:
                # 如果 dateparser 仍然失敗，這代表 LLM 的輸出可能有問題
                logger.warning(f"--- [OllamaTimeParser] dateparser 無法解析 LLM 的輸出: '{simplified_time_str}'。將回傳當前時間。 ---")
                return now
            
        except Exception as e:
            logger.error(f"--- [OllamaTimeParser] 使用 Ollama 解析 '{datetime_str}' 時發生錯誤: {e} ---")
            return now

# 建立 LLMTimeParserService 的一個全域實例
llm_time_parser_service = LLMTimeParserService(model_name="llama3:8b")