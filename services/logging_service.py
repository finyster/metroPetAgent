# services/logging_service.py
import sqlite3
import logging
from datetime import datetime
import json
import config
import os

logger = logging.getLogger(__name__)

class LoggingService:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(config.DATA_DIR, 'interaction_logs.db')
        
        self.db_path = db_path
        self._create_table()
        logger.info(f"--- [LoggingService] 日誌資料庫已準備就緒，存於 {self.db_path} ---")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        """如果資料表不存在，則建立它。"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    thought_process TEXT,
                    tool_name TEXT,
                    tool_input TEXT,
                    tool_output TEXT,
                    final_response TEXT,
                    error_message TEXT
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def log_interaction(self, log_data: dict):
        """將一次完整的互動紀錄寫入資料庫。"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interaction_logs (
                    session_id, timestamp, user_question, thought_process,
                    tool_name, tool_input, tool_output, final_response, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_data.get('session_id'),
                datetime.now().isoformat(),
                log_data.get('user_question'),
                log_data.get('thought_process'),
                log_data.get('tool_name'),
                # 將 tool_input 字典轉換為 JSON 字串以便儲存
                json.dumps(log_data.get('tool_input'), ensure_ascii=False) if log_data.get('tool_input') else None,
                log_data.get('tool_output'),
                log_data.get('final_response'),
                log_data.get('error_message')
            ))
            conn.commit()
            logger.info(f"--- ✅ [LoggingService] 成功記錄一筆互動至資料庫。 ---")
        except Exception as e:
            logger.error(f"--- ❌ [LoggingService] 寫入日誌時發生錯誤: {e} ---", exc_info=True)
        finally:
            conn.close()

# 建立一個全域單例
logging_service = LoggingService()