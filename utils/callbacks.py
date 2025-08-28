# utils/callbacks.py (最終修正版)
from langchain.callbacks.base import AsyncCallbackHandler
from typing import Any, Dict, List, Optional
import logging

# 【★★★ 核心修正 ★★★】
# 將 UUID 從 typing 移出，改為從 uuid 模組直接導入
from uuid import UUID

# 導入我們之前建立的 logging_service
from services.logging_service import logging_service

logger = logging.getLogger(__name__)

class DatabaseCallbackHandler(AsyncCallbackHandler):
    """一個將 Agent 執行過程記錄到資料庫的回呼處理器。"""
    
    def __init__(self, user_question: str, session_id: Optional[str] = None):
        self.user_question = user_question
        self.session_id = session_id
        # 用來暫存這次互動的所有資訊
        self.log_data = {
            "user_question": self.user_question,
            "session_id": self.session_id,
        }

    async def on_agent_action(self, action: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """當 Agent 決定要執行一個工具時觸發。"""
        logger.info(f"Callback: Agent Action - Tool: {action.tool}, Input: {action.tool_input}")
        self.log_data['tool_name'] = action.tool
        self.log_data['tool_input'] = action.tool_input
        # 記錄 Agent 的思考過程
        if 'log' in action and action.log:
            self.log_data['thought_process'] = action.log.strip()

    async def on_tool_end(self, output: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """當工具執行結束時觸發。"""
        logger.info(f"Callback: Tool End - Output: {output[:200]}...") # 只記錄部分輸出
        self.log_data['tool_output'] = output

    async def on_agent_finish(self, finish: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """當 Agent 完成所有工作，準備回傳最終答案時觸發。"""
        logger.info(f"Callback: Agent Finish - Response: {finish.return_values.get('output')}")
        self.log_data['final_response'] = finish.return_values.get('output')
        
        # 在這裡，我們將收集到的所有資訊一次性寫入資料庫
        logging_service.log_interaction(self.log_data)

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """當 Agent 執行鏈中發生任何錯誤時觸發。"""
        logger.error(f"Callback: Chain Error - {error}")
        self.log_data['error_message'] = str(error)
        
        # 即使發生錯誤，也要將已收集到的資訊寫入資料庫，以便分析
        logging_service.log_interaction(self.log_data)