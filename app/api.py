# app/api.py (最終 Conversation Summary Memory 增強版 + 日誌回呼功能)

from fastapi import APIRouter, HTTPException
import logging
from typing import Any, Dict
import uuid # 【★★★ 新增匯入 ★★★】 用於生成獨立的 session ID

# 從 schemas.py 匯入 API 的請求與回應模型
from app.schemas import ChatRequest, ChatResponse, ChatHistory

# --- 核心模組匯入 ---
# 導入 Agent 的核心組件
from agent.agent import agent, all_tools, get_language_instruction
# 導入 LangChain 的進階記憶體模組
from langchain.memory import ConversationSummaryBufferMemory
# 導入 Agent 執行器
from langchain.agents import AgentExecutor
# 導入 LLM，用於建立摘要
from langchain_groq import ChatGroq
# 導入設定檔
import config

# 【★★★ 新增匯入 ★★★】
# 導入我們的資料庫回呼處理器
from utils.callbacks import DatabaseCallbackHandler


logger = logging.getLogger(__name__)
router = APIRouter()

# --- 記憶體與 Agent 執行器設定 ---

# 建立一個專門用於「對話摘要」的、較輕量的 LLM。
# 這可以讓摘要過程更快速、成本更低。
memory_llm = ChatGroq(model="llama3-8b-8192", groq_api_key=config.GROQ_API_KEY)

# 建立一個全域的 AgentExecutor 實例。
# 我們會在每一次的 API 請求中，動態地為它 "附加" 一個獨立的記憶體物件。
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True, # 在後台顯示 Agent 的詳細思考過程，方便除錯
    handle_parsing_errors="抱歉，我好像有點理解錯誤，可以請您換個方式問我嗎？"
)

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    處理聊天請求的核心 API 端點（已升級為 Conversation Summary Memory）。
    """
    try:
        # --- 【★★★ 新增邏輯：建立日誌回呼 ★★★】 ---
        session_id = str(uuid.uuid4()) # 為這次對話生成一個獨一無二的 ID
        db_callback = DatabaseCallbackHandler(
            user_question=request.message,
            session_id=session_id
        )
        
        # --- 步驟 1: 為本次對話建立並載入「摘要記憶體」 ---
        
        # 建立一個新的 ConversationSummaryBufferMemory 物件。
        # 當歷史紀錄的 Token 總量超過 max_token_limit 時，它會自動呼叫 LLM
        # 將最舊的對話內容「壓縮」成一段摘要，以節省空間。
        memory = ConversationSummaryBufferMemory(
            llm=memory_llm,
            max_token_limit=500,       # 建議的 Token 上限，可容納近期詳細對話 + 遠期摘要
            memory_key="chat_history", # 必須對應 Agent Prompt 中的 MessagesPlaceholder
            input_key="input",         # 必須明確告知 Memory 哪個是使用者的輸入
            return_messages=True
        )
        
        # 將前端傳來的歷史紀錄 "載入" 到這個新的 Memory 物件中，
        # 讓 Agent 能夠 "記住" 之前的對話上下文。
        for item in request.chat_history:
            if item.role == "user":
                memory.chat_memory.add_user_message(item.content)
            elif item.role == "assistant":
                memory.chat_memory.add_ai_message(item.content)

        # 將這個載入了歷史的 memory 物件 "附加" 到我們的 AgentExecutor 上
        agent_executor.memory = memory

        # --- 步驟 2: 準備並執行 Agent ---
        lang_name, lang_instruction = get_language_instruction(request.language)
        
        # 準備傳遞給 Agent 的參數。因為 memory 會自動處理 chat_history，所以這裡非常乾淨。
        input_payload = {
            "input": request.message,
            "language_name": lang_name,
            "language_instruction": lang_instruction
        }
        
        logger.info("🚀 正在執行帶有 Summary Buffer Memory 的 AgentExecutor 流程...")
        
        # 【★★★ 修改 Agent 呼叫 ★★★】
        # 執行 Agent，並將我們建立的 db_callback 作為參數傳入
        result = await agent_executor.ainvoke(
            input_payload,
            config={"callbacks": [db_callback]}
        )
        final_output = result['output']

        # --- 步驟 3: 從 Memory 取回更新後的歷史紀錄並回傳 ---
        
        # 執行完畢後，memory 中已經包含了最新的問與答，
        # 並且可能已經將最舊的訊息「摘要」過了。
        updated_history_from_memory = memory.chat_memory.messages
        
        # 將 LangChain 的 Message 物件轉換為 API 需要的字典格式
        history_dicts = [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in updated_history_from_memory
        ]
        
        return ChatResponse(
            response=final_output,
            chat_history=history_dicts
        )
    except Exception as e:
        logger.error(f"Agent 執行出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")