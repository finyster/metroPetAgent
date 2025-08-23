# app/main.py
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# 核心修改：從這裡匯入 router
from app.api import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

app = FastAPI(
    title="MetroPet AI Agent",
    description="An AI agent for Taipei Metro...",
    version="2.0.0"
)

# 掛載 API 路由
app.include_router(api_router, prefix="/api", tags=["API"])

templates = Jinja2Templates(directory="templates")

# 移除 Pydantic 模型定義 (已搬到 schemas.py)
# 移除 invoke_multilingual_agent 函式 (已搬到 api.py)

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """
    根路由，用於提供主聊天頁面 (index.html)。
    """
    return templates.TemplateResponse("index.html", {"request": request})