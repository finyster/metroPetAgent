# services/first_last_train_time_service.py

import logging
import os
import csv 
from typing import Dict, Any, List, Optional

import config
from .station_service import StationManager 
from utils.exceptions import DataLoadError, StationNotFoundError

logger = logging.getLogger(__name__)

class FirstLastTrainTimeService:
    """
    負責載入和查詢捷運站點的首末班車時刻表。
    數據來源為本地處理過的首末班車 CSV 檔案。
    """
    def __init__(self, data_file_path: str, station_manager: StationManager):
        self._data_file_path = data_file_path
        self._station_manager = station_manager
        self._timetable_data: Dict[str, List[Dict[str, Any]]] = {} 
        self._is_loaded = False
        
        self._load_timetable_data()
        logger.info("FirstLastTrainTimeService initialized.")

    def _load_timetable_data(self) -> None:
        """
        載入首末班車時刻表 CSV 數據。
        嘗試多種編碼以提高兼容性，並清理讀取到的字串。
        """
        if not os.path.exists(self._data_file_path):
            logger.error(f"--- ❌ 首末班車時刻表 CSV 檔案不存在: {self._data_file_path}。請確認路徑或先運行 build_database.py。 ---")
            raise DataLoadError(f"首末班車時刻表檔案不存在: {self._data_file_path}")
        
        # 建議調整編碼順序，優先嘗試繁體中文常用編碼
        encodings_to_try = ['utf-8-sig', 'utf-8', 'big5', 'cp950', 'latin-1'] 
        
        for encoding in encodings_to_try:
            try:
                temp_data: Dict[str, List[Dict[str, Any]]] = {}
                # 使用 with open ... as f:
                with open(self._data_file_path, 'r', encoding=encoding, newline='') as f: # 加上 newline='' 是處理csv的標準做法
                    reader = csv.DictReader(f)
                    
                    if not reader.fieldnames:
                        logger.warning(f"CSV 檔案 {self._data_file_path} 在編碼 '{encoding}' 下沒有找到標頭。跳過此編碼。")
                        continue

                    # (檢查欄位的邏輯可以保留)
                    # ...

                    for row in reader:
                        # ===== ★★★ 核心修正點 ★★★ =====
                        # 對所有從CSV讀取的字串欄位，都進行 .strip() 和 .strip("'") 清理
                        # ==================================
                        station_id = row.get('StationID', '').strip().strip("'")
                        
                        if station_id:
                            # 清理其他所有可能用到的字串欄位
                            line_id = row.get('LineID', '').strip().strip("'")
                            trip_head_sign = row.get('TripHeadSign', '').strip().strip("'")
                            
                            # 注意: 請確認您的CSV檔頭真的是 'DestinationStaionID' (有拼寫錯誤)
                            # 如果是，這裡就要用錯的去讀取
                            destination_station_id = row.get('DestinationStaionID', '').strip().strip("'") 
                            destination_station_name = row.get('DestinationStationName', '').strip().strip("'")
                            first_train_time = row.get('FirstTrainTime', '').strip().strip("'")
                            last_train_time = row.get('LastTrainTime', '').strip().strip("'")
                            service_days_raw = row.get('ServiceDays', '').strip().strip("'")
                            update_time = row.get('UpdateTime', '').strip().strip("'")

                            # (後續處理 service_days 的邏輯不變)
                            service_days_display = service_days_raw 
                            cleaned_parts = service_days_raw.strip("{}").split(',') # 這裡也要注意原始資料是 '"{...}"' 還是 "{...}"
                            numeric_parts = [p.strip() for p in cleaned_parts if p.strip().isdigit()]
                            if len(numeric_parts) >= 7 and all(p == '1' for p in numeric_parts[:7]):
                                service_days_display = "每日行駛"
                            # ... (其他邏輯不變)

                            entry = {
                                "LineID": line_id,
                                "StationID": station_id,
                                "TripHeadSign": trip_head_sign,
                                "DestinationStaionID": destination_station_id, 
                                "DestinationStationName": destination_station_name,
                                "FirstTrainTime": first_train_time,
                                "LastTrainTime": last_train_time,
                                "ServiceDays": service_days_display,
                                "UpdateTime": update_time
                            }
                            if station_id not in temp_data:
                                temp_data[station_id] = []
                            temp_data[station_id].append(entry)
                
                # 如果程式能順利執行到這裡，代表這個 encoding 是正確的
                self._timetable_data = temp_data
                self._is_loaded = True
                logger.info(f"--- ✅ 成功載入 {len(self._timetable_data)} 個站點的首末班車時刻表 CSV 數據 (使用編碼: {encoding})。 ---")
                return # 成功載入後就退出函數

            except UnicodeDecodeError as e:
                logger.warning(f"--- ⚠️ 嘗試使用編碼 '{encoding}' 載入時發生解碼錯誤。嘗試下一個編碼。 ---")
            except Exception as e:
                logger.error(f"--- ❌ 載入首末班車時刻表 CSV 數據時發生未知錯誤 (使用編碼: {encoding}): {e} ---", exc_info=True)
                if encoding == encodings_to_try[-1]: 
                    raise DataLoadError(f"載入首末班車時刻表 CSV 數據失敗: {e}")

        # 如果所有編碼都嘗試失敗
        logger.error(f"--- ❌ 無法使用任何已知編碼載入首末班車時刻表 CSV 檔案: {self._data_file_path}。請檢查檔案。 ---")
        raise DataLoadError(f"無法載入首末班車時刻表 CSV 數據，所有編碼嘗試失敗。")
    
    def get_timetable_for_station(self, station_name: str) -> List[Dict[str, Any]]:
        """
        根據站名查詢其所有方向的首末班車時刻表。
        【★ 新增容錯機制 ★】
        如果 StationManager 第一次找不到，會嘗試使用內部的別名表進行二次查詢。
        """
        if not self._is_loaded:
            logger.error("首末班車時刻表服務尚未載入數據。")
            return []

        # --- 階段一：正常嘗試使用 StationManager ---
        station_ids = self._station_manager.get_station_ids(station_name)
        
        # --- 階段二：如果找不到，啟用後備計畫 (Fallback) ---
        if not station_ids:
            logger.warning(f"--- StationManager 初次查詢 '{station_name}' 失敗，嘗試本地別名解析... ---")
            
            # 【新增】一個小型的、僅限於此服務的別名後備表
            # 鍵是使用者可能的輸入，值是猜測的「官方名稱」
            local_aliases = {
                "台北車站": "臺北車站",
                "北車": "臺北車站",
                "台車": "臺北車站",
                "市政府": "臺北市政府",
                "市府站": "臺北市政府",
                "101": "臺北101/世貿",
                "台北101": "臺北101/世貿"
            }
            
            # 檢查使用者輸入是否在我們的後備別名表中
            official_name_guess = local_aliases.get(station_name.strip())
            
            if official_name_guess:
                logger.info(f"--- 本地別名找到: '{station_name}' -> '{official_name_guess}'。將使用此名稱再次查詢 StationManager。 ---")
                # 【關鍵】用解析出的官方名稱，再次呼叫 StationManager
                station_ids = self._station_manager.get_station_ids(official_name_guess)

        # --- 最終檢查 ---
        # 如果經過所有嘗試後，station_ids 仍然為空，則拋出例外
        if not station_ids:
            logger.warning(f"--- 經過所有嘗試，仍找不到車站「{station_name}」的 StationID。 ---")
            raise StationNotFoundError(f"找不到車站「{station_name}」。")
        
        # --- 原有邏輯：用找到的 station_ids 查詢時刻表 ---
        all_timetables = []
        for s_id in station_ids:
            s_id_clean = s_id.strip() 
            if s_id_clean in self._timetable_data:
                for entry in self._timetable_data[s_id_clean]:
                    formatted_entry = {
                        "direction": entry.get('TripHeadSign', '未知方向'),
                        "line_id": entry.get('LineID', '未知路線'),
                        "destination_station": entry.get('DestinationStationName', '未知目的地'),
                        "first_train_time": entry.get('FirstTrainTime', 'N/A'),
                        "last_train_time": entry.get('LastTrainTime', 'N/A'),
                        "service_days": entry.get('ServiceDays', '未知營運日')
                    }
                    all_timetables.append(formatted_entry)
            else:
                logger.debug(f"車站 ID '{s_id_clean}' 在時刻表數據中沒有找到。")
        
        if not all_timetables:
            # 這種情況通常發生在站點ID正確，但首末班車CSV中沒有該站資料時
            logger.warning(f"查無 '{station_name}' (IDs: {station_ids}) 的首末班車資訊。")
        
        return all_timetables