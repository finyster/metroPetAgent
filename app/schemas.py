# app/schemas.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# 定義對話歷史的單條記錄格式
class ChatHistory(BaseModel):
    role: str
    content: str

# 定義前端發送到後端的請求格式
class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatHistory] = Field(default_factory=list)
    language: Optional[str] = "zh-Hant"

# 定義後端回傳給前端的回應格式
class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, Any]]