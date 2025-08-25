# app/api.py (建議修改版)

from fastapi import APIRouter, HTTPException
import logging
import json
from typing import Any, Dict, List, Optional

from app.schemas import ChatRequest, ChatResponse, ChatHistory
# ✨ 這次我們只需要 agent_executor 和 get_language_instruction
from agent.agent import agent_executor, get_language_instruction 
# 移除了 'agent' 和 'get_user_manual_info' 的導入，因為 executor 會處理

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    處理聊天請求的核心 API 端點（簡化版）。
    """
    try:
        # --- 歷史紀錄控管邏輯 (維持不變) ---
        MAX_HISTORY_TURNS = 10
        history_tuples = [(item.role, item.content) for item in request.chat_history]
        if len(history_tuples) > MAX_HISTORY_TURNS:
            logger.info(f"--- 📜 對話歷史過長 ({len(history_tuples)} > {MAX_HISTORY_TURNS})，進行裁切 ---")
            history_tuples = history_tuples[-MAX_HISTORY_TURNS:]

        # --- ✨✨✨【核心修改：移除攔截邏輯，直接呼叫 AgentExecutor】✨✨✨
        
        lang_name, lang_instruction = get_language_instruction(request.language)
        
        # 直接建立完整的輸入 payload
        input_payload = {
            "input": request.message,
            "chat_history": history_tuples,
            "language_name": lang_name,
            "language_instruction": lang_instruction
        }
        
        logger.info("🚀 正在執行完整的 AgentExecutor 流程...")
        # 直接、統一地由 agent_executor 處理所有請求
        result = await agent_executor.ainvoke(input_payload)
        final_output = result['output']
        
        # --- ✨✨✨【修改結束】✨✨✨

        # --- 更新歷史紀錄與回傳 (維持不變) ---
        updated_history = request.chat_history + [
            ChatHistory(role="user", content=request.message),
            ChatHistory(role="assistant", content=final_output)
        ]
        history_dicts = [item.model_dump() for item in updated_history]
        
        return ChatResponse(
            response=final_output,
            chat_history=history_dicts
        )
    except Exception as e:
        logger.error(f"Agent 執行出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")