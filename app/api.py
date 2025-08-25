# app/api.py (最終修正版 v2)

from fastapi import APIRouter, HTTPException
import logging
import json
from typing import Any, Dict, List, Optional

# 從新的 schemas.py 匯入模型
from app.schemas import ChatRequest, ChatResponse, ChatHistory

# --- ✨✨✨【核心修改 1：匯入更多模組】✨✨✨
# 我們不僅需要執行器，還需要 agent 本身，以及手動呼叫的工具
from agent.agent import agent, agent_executor, get_language_instruction
from agent.function_tools import get_user_manual_info
from langchain_core.agents import AgentAction

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    處理聊天請求的核心 API 端點（已整合手動攔截邏輯）。
    """
    try:
        # --- 歷史紀錄控管邏輯 (維持不變) ---
        MAX_HISTORY_TURNS = 10
        history_tuples = [(item.role, item.content) for item in request.chat_history]
        if len(history_tuples) > MAX_HISTORY_TURNS:
            logger.info(f"--- 📜 對話歷史過長 ({len(history_tuples)} > {MAX_HISTORY_TURNS})，進行裁切 ---")
            history_tuples = history_tuples[-MAX_HISTORY_TURNS:]

        # --- ✨✨✨【全新的、更可靠的捷徑邏輯】✨✨✨
        
        # 步驟 A：組合 Agent 需要的輸入，並加入修正 KeyError 的關鍵一行
        lang_name, lang_instruction = get_language_instruction(request.language)
        input_payload = {
            "input": request.message,
            "chat_history": history_tuples,
            "language_name": lang_name,
            "language_instruction": lang_instruction,
            "intermediate_steps": []  # <<< ✨ 修正 KeyError 的關鍵！
        }

        final_output = "" # 用來存放我們最終要回傳的訊息

        # 步驟 B：只運行 Agent 的第一步，讓 AI "決定" 要用哪個工具，但先 "不要執行"
        agent_decision = agent.invoke(input_payload)

        # 步驟 C：檢查 AI 的決定
        if isinstance(agent_decision, list) and agent_decision and isinstance(agent_decision[0], AgentAction):
            action_to_take = agent_decision[0]
            
            # 步驟 D：如果決定是我們的手冊工具，就手動執行並走捷徑
            if action_to_take.tool == "get_user_manual_info":
                logger.info("🚀 攔截成功！手動執行 get_user_manual_info 並跳過第二次 API 呼叫。")
                
                # 手動執行工具函式
                tool_result_str = get_user_manual_info.invoke(action_to_take.tool_input)
                tool_result_json = json.loads(tool_result_str)
                final_output = tool_result_json.get("message", "抱歉，手冊資訊出錯了。")

        # 步驟 E：如果不是手冊工具 (final_output 還是空的)，才執行完整的 Agent 流程
        if not final_output:
             logger.info("🔄 非手冊工具，執行完整的 AgentExecutor 流程。")
             # 在這裡才允許執行完整的 agent_executor
             # 我們傳入原始的 payload，因為 ainvoke 會自己處理 intermediate_steps
             del input_payload["intermediate_steps"]
             result = await agent_executor.ainvoke(input_payload)
             final_output = result['output']

        # --- ✨✨✨【修正結束】✨✨✨

        # --- 更新歷史紀錄與回傳 (使用 final_output) ---
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