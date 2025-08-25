# build_manual_index.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config
import os
import logging
import re # 導入正規表達式函式庫

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_manual_index():
    logger.info("--- 🚀 開始建立【使用者手冊】向量索引 v2.0 ---")
    
    manual_path = os.path.join(config.DATA_DIR, 'user_manual.md')
    if not os.path.exists(manual_path):
        logger.error(f"❌ 找不到使用者手冊檔案: {manual_path}")
        return

    with open(manual_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- ✨✨✨【核心修正：採用按標題切分的智慧化策略】✨✨✨
    # 使用正規表達式，根據 Markdown 的標題 (#) 來切分文件
    # 這個模式會找到一個標題行，以及直到下一個標題之前的所有內容
    raw_chunks = re.split(r'(^# .*)', content, flags=re.MULTILINE)
    
    chunks = []
    # 將標題和其對應的內容合併成一個完整的知識區塊
    for i in range(1, len(raw_chunks), 2):
        header = raw_chunks[i].strip()
        body = raw_chunks[i+1].strip() if (i+1) < len(raw_chunks) else ""
        full_chunk = f"{header}\n\n{body}".strip()
        chunks.append(full_chunk)
    # --- ✨✨✨【修正結束】✨✨✨

    if not chunks:
        logger.error("❌ 未能從手冊中切分出任何知識區塊，請檢查手冊格式。")
        return

    logger.info(f"✅ 成功將手冊切分成 {len(chunks)} 個知識區塊。")

    logger.info("⏳ 正在載入語意模型...")
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    logger.info("⏳ 正在將知識區塊編碼為向量...")
    chunk_embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')

    embedding_dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(chunk_embeddings)

    faiss_index_path = os.path.join(config.DATA_DIR, 'manual_vector.index')
    chunks_path = os.path.join(config.DATA_DIR, 'manual_chunks.json')

    faiss.write_index(index, faiss_index_path)
    import json
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"--- 🎉 成功！使用者手冊向量索引已儲存。 ---")

if __name__ == "__main__":
    build_manual_index()