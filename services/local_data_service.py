# services/local_data_service.py (簡化版)

import json
import config

class LocalDataManager:
    def __init__(self):
        print("--- [LocalData] 正在載入所有本地資料庫... ---")
        self.fares = self._load_json(config.FARE_DATA_PATH, "票價")
        self.facilities = self._load_json(config.FACILITIES_DATA_PATH, "設施")
        self.exits = self._load_json(config.EXIT_DATA_PATH, "出口")
        # 我們直接讓 station_map 也可以從這裡存取，方便工具使用
        self.stations = self._load_json(config.STATION_DATA_PATH, "站點")
        self.user_manual = self._load_json(config.USER_MANUAL_PATH, "使用者手冊")
        self.food_map = self._load_json(config.FOOD_DATA_PATH, "美食地圖")
        self.car_exit_map = self._load_json(config.CAR_EXIT_DATA_PATH, "車廂出口對應")
        print("--- ✅ [LocalData] 所有資料庫載入完成。 ---")

    def _load_json(self, path: str, data_name: str) -> dict:
        """一個健壯的 JSON 載入函式。"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"--- ✅ [LocalData] {data_name}資料庫已載入，共 {len(data)} 筆。")
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"--- ❌ [LocalData] 警告：載入 {data_name} 資料檔案 {path} 失敗: {e} ---")
            return {}

# 建立 LocalDataManager 的單一實例，讓所有工具都能共享已載入的資料
local_data_manager = LocalDataManager()