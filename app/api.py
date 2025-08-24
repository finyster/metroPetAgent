# app/api.py
from fastapi import APIRouter, HTTPException
import logging
from typing import Any, Dict, List, Optional

# 從新的 schemas.py 匯入模型
from app.schemas import ChatRequest, ChatResponse, ChatHistory
# 從 agent 模組直接匯入 agent 執行器
from agent.agent import agent_executor, get_language_instruction

logger = logging.getLogger(__name__)
router = APIRouter()

# 將 invoke_multilingual_agent 函式搬到這裡，因為它是 API 的核心邏輯
async def invoke_multilingual_agent(user_input: str, history: list, lang_code: str):
    lang_name, lang_instruction = get_language_instruction(lang_code)
    input_payload = {
        "input": user_input,
        "chat_history": history,
        "language_name": lang_name,
        "language_instruction": lang_instruction
    }
    return await agent_executor.ainvoke(input_payload)

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    處理聊天請求的核心 API 端點。
    """
    try:
        # --- ✨ 核心修改：加入歷史紀錄控管邏輯 ✨ ---
        
        # 1. 設定一個合理的對話歷史長度上限 (例如保留最近的 20 則訊息)
        MAX_HISTORY_TURNS = 20
        
        # 2. 轉換歷史紀錄
        history_tuples = [(item.role, item.content) for item in request.chat_history]
        
        # 3. 實作滑動視窗：如果歷史紀錄超過上限，只取用最新的部分
        if len(history_tuples) > MAX_HISTORY_TURNS:
            logger.info(f"--- 📜 對話歷史過長 ({len(history_tuples)} > {MAX_HISTORY_TURNS})，進行裁切 ---")
            history_tuples = history_tuples[-MAX_HISTORY_TURNS:]

        # --- ✨ 修改結束 ✨ ---

        result = await invoke_multilingual_agent(
            user_input=request.message,
            history=history_tuples, # 使用可能經過裁切的 history_tuples
            lang_code=request.language
        )
        
        updated_history = request.chat_history + [
            ChatHistory(role="user", content=request.message),
            ChatHistory(role="assistant", content=result['output'])
        ]
        history_dicts = [item.model_dump() for item in updated_history]
        
        return ChatResponse(
            response=result['output'],
            chat_history=history_dicts
        )
    except Exception as e:
        logger.error(f"Agent 執行出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")