# services/realtime_mrt_service.py
import json
import os
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import faiss
import numpy as np
import uuid

# 假設這些是您的其他依賴
# 您原本使用的 metro_soap_service.py
from services.metro_soap_service import MetroSoapService 
# 您原本的 station_service.py
from services.station_service import StationManager 
from utils.time_parser import parse_countdown_to_seconds

logger = logging.getLogger(__name__)

# Define constants for data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "realtime_train_db.json")
DEFAULT_INDEX_PATH = os.path.join(DATA_DIR, "station_index.faiss")

class RealtimeMRTService:
    def __init__(self, metro_soap_api: MetroSoapService, station_manager: StationManager, update_interval_seconds: int = 15, db_path: str = DEFAULT_DB_PATH, index_path: str = DEFAULT_INDEX_PATH):
        self.metro_soap_api = metro_soap_api
        self.station_manager = station_manager
        self.update_interval_seconds = update_interval_seconds
        self.db_path = db_path
        self.index_path = index_path
        self._cached_train_info: List[Dict[str, Any]] = []
        self._cache_timestamp: Optional[datetime] = None
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._station_index = None
        self._station_names_list: List[str] = []

        self._load_local_db_and_update_sync()
        self._init_faiss_index()
        logger.info(f"--- RealtimeMRTService 初始化，數據每 {update_interval_seconds} 秒刷新，DB 存於 {db_path} ---")

    def _load_local_db_and_update_sync(self):
        """從本地 DB 載入數據，並嘗試立即同步更新。"""
        self._load_local_db()
        # 如果快取沒有數據或已過期，則立即同步更新
        if not self._cached_train_info or (self._cache_timestamp and (datetime.now() - self._cache_timestamp).total_seconds() > self.update_interval_seconds):
            logger.info("--- 🔄 本地數據過期或為空，正在進行同步更新... ---")
            self.update_cache_sync()
        else:
            logger.info("--- ✅ 本地數據仍然有效，無需同步更新。 ---")

    def update_cache_sync(self):
        """同步方式刷新緩存，用於應急。"""
        try:
            all_track_info = self.metro_soap_api.get_realtime_track_info()
            if all_track_info:
                self._cached_train_info = all_track_info
                self._cache_timestamp = datetime.now()
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                with open(self.db_path, 'w', encoding='utf-8') as f:
                    # 每次寫入都是全新的數據，舊數據會被覆蓋，實現「清洗」
                    json.dump({"timestamp": self._cache_timestamp.isoformat(), "trains": all_track_info}, f, ensure_ascii=False, indent=2)
                logger.info(f"--- ✅ 同步緩存刷新完成，共 {len(all_track_info)} 筆列車資訊 ---")
                return True
            else:
                logger.warning("--- ⚠️ 未從 Metro API 獲取到任何列車資訊 (同步呼叫) ---")
        except Exception as e:
            logger.error(f"--- ❌ 同步刷新緩存時發生錯誤: {e} ---", exc_info=True)
        return False

    def _load_local_db(self):
        """從本地 JSON 資料庫載入數據。"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cached_train_info = data.get("trains", [])
                    timestamp_str = data.get("timestamp")
                    if timestamp_str:
                        self._cache_timestamp = datetime.fromisoformat(timestamp_str)
                    logger.info(f"--- ✅ 從 {self.db_path} 載入本地列車緩存 (共 {len(self._cached_train_info)} 筆) ---")
            except Exception as e:
                logger.error(f"--- ❌ 載入本地列車緩存時發生錯誤: {e} ---", exc_info=True)
    
    def _init_faiss_index(self):
        """初始化 FAISS 索引，基於站名嵌入向量。"""
        self._station_names_list = list(self.station_manager.station_map.keys())
        
        if not self._station_names_list:
            logger.warning("--- ⚠️ StationManager 中沒有可用的站名，FAISS 索引無法初始化 ---")
            return

        embedding_dim = 128
        if os.path.exists(self.index_path):
            try:
                self._station_index = faiss.read_index(self.index_path)
                logger.info(f"--- ✅ 從 {self.index_path} 載入 FAISS 索引 ---")
            except Exception as e:
                logger.error(f"--- ❌ 載入 FAISS 索引時發生錯誤: {e}，將重新創建 ---", exc_info=True)
                self._station_index = None

        if self._station_index is None:
            logger.info("--- 🔄 正在創建新的 FAISS 索引 ---")
            station_embeddings = np.array([
                np.frombuffer(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()).bytes, dtype=np.uint8)[:embedding_dim].astype('float32') / 255.0
                for name in self._station_names_list
            ])
            
            # 確保嵌入維度正確
            if station_embeddings.shape[1] < embedding_dim:
                padded_embeddings = np.zeros((station_embeddings.shape[0], embedding_dim), dtype='float32')
                padded_embeddings[:, :station_embeddings.shape[1]] = station_embeddings
                station_embeddings = padded_embeddings
            elif station_embeddings.shape[1] > embedding_dim:
                station_embeddings = station_embeddings[:, :embedding_dim]

            if station_embeddings.shape[1] != embedding_dim:
                logger.error(f"生成的站名嵌入維度不正確: {station_embeddings.shape[1]} vs {embedding_dim}")
                return

            self._station_index = faiss.IndexFlatL2(embedding_dim)
            self._station_index.add(station_embeddings)
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                faiss.write_index(self._station_index, self.index_path)
            except Exception as e:
                logger.error(f"--- ❌ 保存 FAISS 索引時發生錯誤: {e} ---", exc_info=True)
                self._station_index = None

    def start_update_thread(self):
        if not self._is_running:
            self._stop_event.clear()
            self._update_thread = threading.Thread(target=self._periodic_update_cache, daemon=True)
            self._update_thread.start()
            self._is_running = True
            logger.info("--- RealtimeMRTService 更新線程已啟動 ---")

    def stop_update_thread(self):
        if self._is_running:
            self._stop_event.set()
            if self._update_thread:
                self._update_thread.join(timeout=self.update_interval_seconds + 5)
            self._is_running = False
            logger.info("--- RealtimeMRTService 更新線程已停止 ---")

    def _periodic_update_cache(self):
        while not self._stop_event.is_set():
            logger.info("--- RealtimeMRTService: 正在刷新即時列車資訊緩存... ---")
            try:
                all_track_info = self.metro_soap_api.get_realtime_track_info()
                if all_track_info:
                    self._cached_train_info = all_track_info
                    self._cache_timestamp = datetime.now()
                    os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                    with open(self.db_path, 'w', encoding='utf-8') as f:
                        json.dump({"timestamp": self._cache_timestamp.isoformat(), "trains": all_track_info}, f, ensure_ascii=False, indent=2)
                    logger.info(f"--- ✅ 緩存刷新完成，共 {len(all_track_info)} 筆列車資訊，存至 {self.db_path} ---")
                else:
                    logger.warning("--- ⚠️ 未從 Metro API 獲取到任何列車資訊 ---")
            except Exception as e:
                logger.error(f"--- ❌ 刷新緩存時發生錯誤: {e} ---", exc_info=True)
            self._stop_event.wait(self.update_interval_seconds)

    def get_realtime_train_info(self) -> List[Dict[str, Any]]:
        """
        獲取最新的即時列車資訊。
        如果緩存過期，則會嘗試同步刷新。如果同步刷新失敗，會從本地 DB 讀取最近的數據。
        """
        # 檢查緩存是否仍有效
        if self._cache_timestamp and (datetime.now() - self._cache_timestamp).total_seconds() < self.update_interval_seconds:
            logger.info("--- ✅ 列車緩存仍然有效 ---")
            return self._cached_train_info
        
        # 緩存過期，立即嘗試同步刷新
        logger.warning("--- ⚠️ 列車緩存數據已過期，正在嘗試同步獲取最新資料... ---")
        if self.update_cache_sync():
            return self._cached_train_info
        else:
            # 如果同步刷新失敗，則嘗試從本地 DB 讀取最近的數據（即使可能稍舊）
            self._load_local_db()
            if self._cached_train_info:
                logger.warning("--- ❌ 同步刷新失敗，回傳過期但尚可用的本地 DB 資料。---")
                return self._cached_train_info
            
        logger.error("--- ❌ 無法獲取任何列車資訊，請檢查 API 連線與設定。 ---")
        return []

    def get_next_train_info(self, target_station_official_name: str, target_direction_normalized_list: List[str]) -> List[Dict[str, Any]]:
        """
        獲取針對特定車站和方向篩選後的列車資訊。
        """
        all_train_info = self.get_realtime_train_info()
        if not all_train_info:
            logger.warning("--- ⚠️ 無法獲取列車資訊以篩選下一班列車。 ---")
            return []
        
        target_station_normalized = self.station_manager._normalize_name_for_map(target_station_official_name)
        candidate_trains = []

        for train in all_train_info:
            train_current_station_raw = train.get('StationName')
            train_destination_raw = train.get('DestinationName')
            countdown_str = train.get('CountDown', '')

            train_current_station_normalized = self.station_manager._normalize_name_for_map(train_current_station_raw)
            train_destination_normalized = self.station_manager._normalize_name_for_map(train_destination_raw)

            countdown_seconds = parse_countdown_to_seconds(countdown_str)
            if countdown_seconds == float('inf'):
                continue

            if train_destination_normalized in target_direction_normalized_list:
                # 簡化判斷邏輯：只檢查列車當前站是否為目標站或目標站的鄰近站
                # 由於 MetroSoapService API 的資料格式可能與 TDX API 不同，我們假設
                # 這裡的 "StationName" 是指列車目前所在的車站。
                if train_current_station_normalized == target_station_normalized:
                    candidate_trains.append(train)

        candidate_trains.sort(key=lambda x: parse_countdown_to_seconds(x.get('CountDown', '')))
        
        return candidate_trains

    def search_station(self, query: str) -> Optional[str]:
        """
        【重要】優先使用精確匹配，失敗後才使用 FAISS 模糊匹配。
        """
        # 步驟1: 嘗試精確匹配（包含別名）
        resolved_name_by_manager = self.station_manager.resolve_station_alias(query)
        if self.station_manager.get_station_ids(resolved_name_by_manager):
            logger.info(f"--- 精確匹配 '{query}' 成功，解析為 '{resolved_name_by_manager}' ---")
            return resolved_name_by_manager

        # 步驟2: 如果精確匹配失敗，才使用 FAISS 進行模糊匹配
        if self._station_index is None or not self._station_names_list:
            logger.warning("--- ⚠️ FAISS 索引未初始化或站名列表為空，無法進行模糊搜索 ---")
            return None

        embedding_dim = self._station_index.d
        query_embedding_bytes = uuid.uuid5(uuid.NAMESPACE_DNS, query.lower()).bytes
        query_embedding = np.zeros(embedding_dim, dtype='float32')
        temp_embedding = np.frombuffer(query_embedding_bytes, dtype=np.uint8).astype('float32') / 255.0
        query_embedding[:min(embedding_dim, temp_embedding.shape[0])] = temp_embedding[:min(embedding_dim, temp_embedding.shape[0])]
        query_embedding = query_embedding.reshape(1, -1)

        try:
            # 修正: 調整 L2 距離閾值
            faiss_l2_distance_threshold = 1.0
            distances, indices = self._station_index.search(query_embedding, k=1)
            
            if distances[0][0] <= faiss_l2_distance_threshold:
                resolved_name_by_faiss = self._station_names_list[indices[0][0]]
                logger.info(f"--- FAISS 模糊搜索 '{query}' 成功，解析為 '{resolved_name_by_faiss}' (L2距離: {distances[0][0]:.4f}) ---")
                return resolved_name_by_faiss
            else:
                logger.info(f"--- FAISS 搜索 '{query}' 未找到高相似度結果 (L2距離: {distances[0][0]:.4f}) ---")
        except Exception as e:
            logger.error(f"--- ❌ FAISS 搜索時發生錯誤: {e} ---", exc_info=True)
        
        return None

    def resolve_train_terminus(self, start_station_name: str, intermediate_destination: str) -> List[str]:
        """
        根據起點和中間站點，推導出可能的列車終點站。
        """
        # 使用 StationManager 的 resolve_direction 方法，這個方法已經包含了所有邏輯
        # 它會將「往淡水」這類的指令解析為實際的終點站名稱，並處理別名
        resolved_terminus = self.station_manager.resolve_direction(start_station_name, intermediate_destination)
        
        if not resolved_terminus:
            logger.warning(f"--- ⚠️ 無法從 '{start_station_name}' 往 '{intermediate_destination}' 推導出可能的終點站 ---")
        
        return resolved_terminus
    
    # 這是為了配合工具函式而新增的入口方法
    def get_arrival_info(self, station_name: str, destination_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        獲取特定站點的列車到站資訊。
        
        Args:
            station_name (str): 使用者查詢的站點名稱，可能為別名。
            destination_name (Optional[str]): 列車行駛的終點站名稱，可能為別名或模糊方向。
            
        Returns:
            List[Dict[str, Any]]: 包含列車到站資訊的列表，或一個空列表。
        """
        # 1. 解析站名：將使用者輸入的站名解析為標準化的官方名稱
        resolved_station_name = self.search_station(station_name)
        if not resolved_station_name:
            logger.error(f"無法解析查詢站點名稱: '{station_name}'")
            return []

        # 2. 解析方向：找出所有可能的終點站名稱
        if destination_name:
            resolved_directions = self.resolve_train_terminus(resolved_station_name, destination_name)
            if not resolved_directions:
                logger.warning(f"無法解析從 '{station_name}' 往 '{destination_name}' 的方向。")
                return []
        else:
            # 如果沒有指定方向，則獲取該站點所有線路的所有終點站
            resolved_directions = self.station_manager.get_terminal_stations_for(resolved_station_name)
            
        # 3. 根據解析後的站名和方向，從快取中篩選列車資訊
        filtered_data = self.get_next_train_info(resolved_station_name, resolved_directions)
        
        return filtered_data