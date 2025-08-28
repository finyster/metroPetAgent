# services/llm_network_service.py (最終穩健版)
import logging
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
import config
import random

logger = logging.getLogger(__name__)

class LLMNetworkService:
    def __init__(self, model_name: str = "llama3.1:latest"):
        logger.info(f"--- [LLMNetworkService RAG Final] 正在初始化... (模型: {model_name}) ---")
        self.is_ready = False
        try:
            # 1. 載入 RAG 元件
            self.retrieval_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            self.knowledge_index = faiss.read_index(config.NETWORK_KNOWLEDGE_INDEX_PATH)
            with open(config.NETWORK_KNOWLEDGE_CHUNKS_PATH, 'r', encoding='utf-8') as f:
                self.knowledge_chunks = json.load(f)
            
            # 2. 載入生成模型 (Ollama)
            self.generation_model = ChatOllama(model=model_name, temperature=0.1)
            
            self.is_ready = True
            logger.info("--- ✅ [LLMNetworkService RAG Final] 已就緒。 ---")
        except Exception as e:
            logger.error(f"--- ❌ [LLMNetworkService RAG Final] 初始化失敗: {e} ---")
            logger.error("--- 👉 請先執行 `python build_network_knowledge_base.py` ---")

    def _retrieve_context(self, user_question: str, top_k: int = 5) -> str:
        """從向量知識庫中，檢索與問題最相關的知識卡片。"""
        query_embedding = self.retrieval_model.encode([user_question])
        query_embedding = np.array(query_embedding).astype('float32')
        _, indices = self.knowledge_index.search(query_embedding, top_k)
        
        retrieved_chunks = [self.knowledge_chunks[i] for i in indices[0]]
        logger.info(f"--- [RAG] 檢索到 {len(retrieved_chunks)} 筆相關資料。")
        return "\n".join(retrieved_chunks)

    def answer_query(self, user_question: str) -> str:
        """執行 RAG 流程：先檢索，後生成。"""
        if not self.is_ready:
            return "抱歉，路網問答服務尚未準備好。"

        # 步驟 A: 直接、純粹地進行 RAG 檢索
        relevant_context = self._retrieve_context(user_question)

        # 步驟 B: 將檢索結果與一個更聰明的 Prompt 交給 LLM
        prompt = f"""
        你是一個名為「捷米」的專業台北捷運路網問答助理。你的職責是根據下方提供的「相關資料」，智慧地回答「使用者問題」。

        **行為準則：**
        1.  **分析與回答**: 首先，仔細分析「使用者問題」。然後，在「相關資料」中尋找能直接回答問題的資訊。
        2.  **精準回答**: 如果「相關資料」中包含能直接回答問題的資訊（例如，使用者問A，資料裡有A），請直接、準確地回答。
        3.  **智慧應變**: 如果「使用者問題」很模糊（例如「隨便給我幾個站」），而「相關資料」看起來是一堆不直接相關的範例，請不要列出所有資料，而是從中挑選 2-3 個例子，並用一句話自然地回答，例如「當然！台北捷運有很多車站，像是台北車站、市政府站等等喔！」。
        4.  **誠實原則**: 如果「相關資料」中真的沒有答案，就誠實地回答「根據我手邊的資料，我找不到相關資訊」。

        ---
        **【相關資料】**
        {relevant_context}
        ---

        **【使用者問題】**
        "{user_question}"

        **【你的簡潔回答】**
        """
        try:
            response = self.generation_model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"--- ❌ [LLMNetworkService RAG Final] 生成回答時發生錯誤: {e} ---")
            return "抱歉，我在思考路網問題時遇到了一點困難。"