# build_network_knowledge_base.py
import logging
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import config
from services import service_registry # 導入 service_registry 以獲取 routing_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_network_knowledge_base():
    logger.info("--- 🚀 開始建立【混合策略 RAG 知識庫】 v4.0 ---")
    
    routing_manager = service_registry.get_routing_manager()
    if not routing_manager.is_graph_ready:
        logger.error("--- ❌ RoutingManager 未就緒，無法建立知識庫。 ---")
        return

    # --- 1. 將路網知識「卡片化」 ---
    knowledge_chunks = []
    
    # --- 【策略一：保留以「路線」為主題的知識卡片】 ---
    all_lines_data = routing_manager.list_all_lines().get("lines", [])
    for line in all_lines_data:
        line_name = line['line_name']
        details = routing_manager.get_line_details(line_name)
        stations_str = "、".join(details.get("stations", []))
        knowledge_chunks.append(f"捷運{line_name}的車站包含：{stations_str}。")
    
    logger.info(f"✅ 已生成 {len(all_lines_data)} 張「路線主題」知識卡片。")

    # --- 【策略二：建立以「車站」為主題的綜合知識卡片】 ---
    all_station_names = routing_manager.station_manager.get_all_station_names()
    station_card_count = 0
    for station_name in all_station_names:
        station_facts = []

        # 獲取所屬路線與轉乘資訊
        lines_data = routing_manager.get_lines_for_station(station_name)
        if "error" not in lines_data:
            lines = lines_data.get("lines", [])
            if lines:
                line_names = [line['line_name'] for line in lines]
                station_facts.append(f"{station_name}位於：{'、'.join(line_names)}。")
                if len(lines) > 1:
                    station_facts.append(f"{station_name}是一個轉乘站。")

        # 獲取鄰近車站資訊
        neighbor_data = routing_manager.get_neighbor_stations(station_name)
        if "error" not in neighbor_data:
            neighbors = neighbor_data.get("neighbors", [])
            if neighbors:
                cleaned_neighbors = [n.replace("站", "") for n in neighbors]
                station_facts.append(f"鄰近的車站有：{'、'.join(cleaned_neighbors)}。")
        
        # 將所有關於此車站的事實，組合成一張知識卡片
        if station_facts:
            full_chunk = " ".join(station_facts)
            knowledge_chunks.append(full_chunk)
            station_card_count += 1

    logger.info(f"✅ 已生成 {station_card_count} 張「車站主題」知識卡片。")
    logger.info(f"✅ 知識庫總卡片數量: {len(knowledge_chunks)}。")

    # --- 2. 建立向量索引 (邏輯不變) ---
    logger.info("⏳ 正在載入語意模型...")
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    logger.info("⏳ 正在將知識卡片編碼為向量...")
    chunk_embeddings = model.encode(knowledge_chunks, convert_to_tensor=False, show_progress_bar=True)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')

    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    # --- 3. 儲存索引與知識卡片 (邏輯不變) ---
    faiss.write_index(index, config.NETWORK_KNOWLEDGE_INDEX_PATH)
    with open(config.NETWORK_KNOWLEDGE_CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"--- 🎉 成功！混合策略 RAG 知識庫已建立。 ---")

if __name__ == "__main__":
    build_network_knowledge_base()