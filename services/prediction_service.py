import pandas as pd
import xgboost as xgb
import joblib
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import dateparser

# --- 【新增導入】 ---
import holidays # 導入 holidays 函式庫

# --- 路徑設置 ---
import sys
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVICE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.station_service import StationManager
from services.metro_soap_service import metro_soap_api
import config # 確保 config 檔案存在且可被導入

# --- 配置日誌 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CongestionPredictor:
    def __init__(self, station_manager_instance: StationManager):
        logger.info("--- [Predictor] 正在初始化人流預測服務... ---")
        self.station_manager = station_manager_instance
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.encoders: Dict[str, any] = {}
        self.scalers: Dict[str, any] = {}
        self.feature_columns: Dict[str, list] = {}
        # --- 【新增】初始化台灣假日物件 ---
        self.tw_holidays = holidays.TW() # 初始化台灣假日物件
        self.is_ready = self._load_all_models()

        if self.is_ready:
            logger.info("--- ✅ 預測服務已成功載入模型並準備就緒。 ---")
        else:
            logger.error("--- ❌ 預測服務初始化失敗，部分模型或檔案缺失。 ---")

    def _load_all_models(self) -> bool:
        all_loaded = True
        for line_type in ['high_capacity', 'wenhu']:
            model_path = os.path.join(MODEL_DIR, f'{line_type}_congestion_model.json')
            encoder_path = os.path.join(MODEL_DIR, f'{line_type}_encoder.joblib')
            scaler_path = os.path.join(MODEL_DIR, f'{line_type}_scaler.joblib')
            features_path = os.path.join(MODEL_DIR, f'{line_type}_feature_columns.csv')
            
            if not all(os.path.exists(p) for p in [model_path, encoder_path, scaler_path, features_path]):
                logger.warning(f"--- ⚠️ 在路徑 '{MODEL_DIR}' 中找不到 {line_type} 的模型檔案，請先運行 model_trainer.py。 ---")
                all_loaded = False
                continue
            
            try:
                self.models[line_type] = xgb.XGBClassifier()
                self.models[line_type].load_model(model_path)
                self.encoders[line_type] = joblib.load(encoder_path)
                self.scalers[line_type] = joblib.load(scaler_path)
                # 您的 feature_columns.csv 假設有 'feature' 標頭
                self.feature_columns[line_type] = pd.read_csv(features_path)['feature'].tolist()
                logger.info(f"--- ✅ 已成功從 '{MODEL_DIR}' 載入 {line_type} 模型。 ---")
            except Exception as e:
                logger.error(f"載入 {line_type} 模型時發生錯誤: {e}", exc_info=True)
                all_loaded = False
        return all_loaded

    def _get_line_type_and_id(self, station_name: str) -> Optional[Tuple[str, str]]:
        station_ids = self.station_manager.get_station_ids(station_name)
        if not station_ids:
            logger.warning(f"無法在 StationManager 中找到站名 '{station_name}' 的任何 ID。")
            return None, None
        station_id = station_ids[0]
        if station_id.startswith('BR'):
            return 'wenhu', station_id
        return 'high_capacity', station_id

    def _create_prediction_features(self, station_id: str, line_direction_cid: str, line_type: str, target_datetime: datetime) -> pd.DataFrame:
        """
        根據指定的日期時間，創建模型所需的特徵。
        --- 【最終優化版】此函式現在的邏輯與 model_trainer.py 中的特徵工程完全一致。 ---
        """
        with open(os.path.join(DATA_DIR, 'mrt_station_info.json'), 'r', encoding='utf-8') as f:
            station_info = json.load(f)
        transfer_stations = {sid for info in station_info.values() if isinstance(info, dict) for sid in info.get('station_ids', []) if info.get('is_transfer')}
        
        # 模擬滯後特徵 (與訓練腳本一致的簡易邏輯)
        lag_5min_congestion = 0.0
        lag_1hr_congestion = 0.0
        if target_datetime.weekday() < 5 and target_datetime.hour in [7, 8, 17, 18]:
            lag_5min_congestion = 1.5 
            lag_1hr_congestion = 2.0  
        elif target_datetime.weekday() >= 5:
            lag_5min_congestion = 1.0 
            lag_1hr_congestion = 1.0
        if target_datetime.hour in [21, 22, 23, 0, 1, 2, 3, 4, 5]:
            lag_5min_congestion = 0.5
            lag_1hr_congestion = 0.5
        
        num_cars = 4 if line_type == 'wenhu' else 6
        records = []
        for car_num in range(1, num_cars + 1):
            records.append({
                'station_id': station_id,
                'line_direction_cid': line_direction_cid, # 這裡保持字串
                'hour': target_datetime.hour,
                'minute': target_datetime.minute,
                'day_of_week': target_datetime.weekday(),
                'month': target_datetime.month,
                'year': target_datetime.year,
                'is_weekend': int(target_datetime.weekday() >= 5),
                'is_morning_peak': int(7 <= target_datetime.hour <= 9),
                'is_evening_peak': int(17 <= target_datetime.hour <= 19),
                'is_peak_hour': int(target_datetime.hour in [7, 8, 17, 18, 19]),
                'is_holiday': int(target_datetime.date() in self.tw_holidays),
                'is_transfer_station': int(station_id in transfer_stations),
                'car_number': car_num,
                'lag_5min_congestion': lag_5min_congestion,
                'lag_1hr_congestion': lag_1hr_congestion
            })
        
        df_raw = pd.DataFrame(records)
        
        encoder = self.encoders[line_type]
        # 關鍵修正：確保這裡的類別特徵列表與訓練時一致
        categorical_features = ['station_id', 'line_direction_cid']
        encoded_data = encoder.transform(df_raw[categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
        
        # 關鍵修正：確保這裡的數值特徵列表與訓練時一致
        numeric_features = [
            'hour', 'minute', 'day_of_week', 'month', 'year', 'is_weekend', 
            'is_morning_peak', 'is_evening_peak', 'is_peak_hour', 'is_holiday', 
            'is_transfer_station', 'car_number', 'lag_5min_congestion', 'lag_1hr_congestion'
        ]

        final_df = pd.concat([df_raw[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        
        scaler = self.scalers[line_type]
        final_df[numeric_features] = scaler.transform(final_df[numeric_features])
        
        # 【最終保險】使用訓練時儲存的欄位順序，確保特徵的名稱、數量、順序完全一致
        final_df = final_df.reindex(columns=self.feature_columns[line_type], fill_value=0)
        
        return final_df

    def predict_for_station(self, station_name: str, direction: str, target_datetime: datetime) -> Dict[str, Any]:
        """
        為指定車站和方向提供通用的車廂擁擠度預測。
        """
        if not self.is_ready:
            return {"error": "預測服務尚未準備就緒，請檢查模型檔案是否存在。"}

        line_type, station_id = self._get_line_type_and_id(station_name)
        if not line_type:
            return {"error": f"無法識別車站 '{station_name}'，請確認站名是否正確。"}
            
        direction_map = {
            "南港展覽館": 1, "動物園": 1, "迴龍": 1, "蘆洲": 1, "淡水": 1, "北投": 1,
            "頂埔": 2, "象山": 2, "大安": 2, "南勢角": 2, "新店": 2, "台電大樓": 2, "板橋": 2
        }
        # 【優化】清理方向字串，使其能匹配 direction_map 的鍵
        normalized_direction = direction.replace("往", "").replace("站", "")
        line_direction_cid = direction_map.get(normalized_direction)

        if line_direction_cid is None:
            logger.error(f"--- ❌ 無法在 direction_map 中找到方向 '{normalized_direction}' 的對應 CID ---")
            return {"error": f"無法識別方向 '{direction}'，請確認方向是否正確。"}

        logger.info(f"開始為車站 '{station_name}' (ID: {station_id}, 方向: {line_direction_cid}) 於 {target_datetime.strftime('%Y-%m-%d %H:%M')} 進行預測...")
        
        try:
            # 【優化】在呼叫前將 CID 轉為字串，以匹配 one-hot encoder 的期望輸入
            line_direction_cid_str = str(line_direction_cid)
            X_pred = self._create_prediction_features(station_id, line_direction_cid_str, line_type, target_datetime)
            
            model = self.models[line_type]
            predictions = model.predict(X_pred)
            
            congestion_map = {0: "舒適", 1: "正常", 2: "略多", 3: "擁擠"}
            results = []
            for i, pred_class in enumerate(predictions):
                level = int(pred_class)
                # 後處理：確保尖峰時段的預測更貼近現實
                if target_datetime.weekday() < 5 and target_datetime.hour in [7, 8, 17, 18]:
                    if level == 0: level = 1
                
                results.append({
                    "car_number": i + 1,
                    "congestion_level": level + 1,
                    "congestion_text": congestion_map.get(level, "未知")
                })

            return {
                "station_name": station_name,
                "direction": direction,
                "prediction_time": target_datetime.isoformat(),
                "congestion_by_car": results
            }
        except Exception as e:
            logger.error(f"為 '{station_name}' 進行預測時發生錯誤: {e}", exc_info=True)
            return {"error": f"預測時發生內部錯誤，請檢查日誌。"}

    def predict_next_train_congestion(self, station_name: str, direction: str) -> Dict[str, Any]:
        """
        結合即時列車資訊與擁擠度預測模型，為使用者提供即將到站列車的預測結果。
        """
        if not self.is_ready:
            return {"error": "預測服務尚未準備就緒，請檢查模型檔案是否存在。"}

        logger.info(f"--- 🚀 正在從 Metro API 獲取即時列車資訊以查找車站 '{station_name}' 往 '{direction}' 方向 ---")
        try:
            all_train_info = metro_soap_api.get_realtime_track_info() 
        except Exception as e:
            logger.error(f"獲取即時列車資訊時發生錯誤: {e}", exc_info=True)
            return {"error": "無法從 Metro API 獲取即時列車資訊，請檢查服務連線。"}

        congestion_prediction_for_station = self.predict_for_station(station_name, direction, target_datetime=datetime.now())

        if "error" in congestion_prediction_for_station:
            return {"error": congestion_prediction_for_station["error"]}

        relevant_trains = []
        if all_train_info:
            for train in all_train_info:
                if train and 'DestinationName' in train and direction in train.get('DestinationName', ''):
                    relevant_trains.append(train)

            def parse_countdown_to_seconds(countdown_str):
                if countdown_str == '列車進站':
                    return 0
                if '分' in countdown_str and '秒' in countdown_str:
                    parts = countdown_str.replace(' 分鐘 ', ' ').replace(' 秒', '').split(' ')
                    if len(parts) == 2:
                        try:
                            minutes = int(parts[0])
                            seconds = int(parts[1])
                            return minutes * 60 + seconds
                        except ValueError:
                            return float('inf')
                return float('inf')

            relevant_trains.sort(key=lambda x: parse_countdown_to_seconds(x.get('CountDown', '未知')))

        return {
            "station_name": station_name,
            "direction": direction,
            "prediction_time": datetime.now().isoformat(),
            "relevant_trains_info": relevant_trains,
            "congestion_prediction_for_station": congestion_prediction_for_station
        }