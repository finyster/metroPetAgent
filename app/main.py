# app/main.py
"""
======================================================================
|                MetroPet AI Agent - FastAPI Entrypoint                |
|                   (採用 finyster 分支的強化架構)                   |
======================================================================
此檔案是 FastAPI 應用程式的主要入口點。
負責處理 HTTP 請求，並將使用者的訊息傳遞給 AI Agent。
"""

# ---------------------------------------------------------------------
# 1. 核心模組匯入 (Core Module Imports)
# ---------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# 導入我們最終合併好的 Agent 執行器和多語言輔助函式
from agent.agent import agent_executor, get_language_instruction

# ---------------------------------------------------------------------
# 2. 應用程式初始化與設定 (App Initialization & Setup)
# ---------------------------------------------------------------------

# 在所有程式碼執行前，設定日誌的基礎配置，這會讓 Uvicorn 的日誌輸出更清晰、更專業。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 強制覆蓋 Uvicorn 的預設配置
)

# 初始化 FastAPI 應用
app = FastAPI(
    title="MetroPet AI Agent",
    description="An AI agent for Taipei Metro, integrating routing, real-time data, and lifestyle information.",
    version="2.0.0" # Version updated to reflect the merge
)

# 設定模板目錄，用於渲染 HTML 頁面
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------
# 3. 資料模型定義 (Pydantic Data Models)
# ---------------------------------------------------------------------

# 定義對話歷史的單條記錄格式
class ChatHistory(BaseModel):
    role: str
    content: str

# 定義前端發送到後端的請求格式
class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatHistory] = Field(default_factory=list)
    # 增加語言欄位，並設定預設值，完美對應前端的多語言功能
    language: Optional[str] = "zh-Hant"

# 定義後端回傳給前端的回應格式
class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, Any]]


# ---------------------------------------------------------------------
# 4. API 路由定義 (API Route Definitions)
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """
    根路由，用於提供主聊天頁面 (index.html)。
    """
    return templates.TemplateResponse("index.html", {"request": request})


async def invoke_multilingual_agent(user_input: str, history: list, lang_code: str):
    """
    包裝原始的 agent_executor.ainvoke，動態加入多語言指令。
    這是連接前端語言選擇和後端 Agent Prompt 的橋樑。
    """
    # 1. 根據 lang_code 獲取語言名稱和對應的指令
    lang_name, lang_instruction = get_language_instruction(lang_code)

    # 2. 準備傳遞給 agent_executor 的完整 payload
    input_payload = {
        "input": user_input,
        "chat_history": history,
        "language_name": lang_name,
        "language_instruction": lang_instruction
    }

    # 3. 非同步呼叫 Agent 執行器
    return await agent_executor.ainvoke(input_payload)


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    處理聊天請求的核心 API 端點。
    接收使用者訊息、對話歷史和語言，呼叫 Agent，並返回結果。
    """
    try:
        # 將 Pydantic 模型列表轉換為 LangChain Agent 需要的 (role, content) 元組列表
        history_tuples = [(item.role, item.content) for item in request.chat_history]

        # 呼叫我們新建的包裝函式，傳入所有必要資訊
        result = await invoke_multilingual_agent(
            user_input=request.message,
            history=history_tuples,
            lang_code=request.language
        )

        # 在返回前，更新對話歷史，將最新的問與答加入
        updated_history = request.chat_history + [
            ChatHistory(role="user", content=request.message),
            ChatHistory(role="assistant", content=result['output'])
        ]

        # 將 Pydantic 模型轉回字典列表以便 JSON 序列化
        # 使用 .model_dump() 是 Pydantic V2 的推薦作法，取代舊的 .dict()
        history_dicts = [item.model_dump() for item in updated_history]

        return ChatResponse(
            response=result['output'],
            chat_history=history_dicts
        )
    except Exception as e:
        logging.error(f"Agent 執行出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="抱歉，我現在有點問題，請稍後再試。")