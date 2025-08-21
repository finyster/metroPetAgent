# data_collector.py

import pandas as pd
import os
import logging
from datetime import datetime
# 假設您的 service 檔案位於 services/metro_soap_service.py
from services.metro_soap_service import metro_soap_api
import time

# --- (日誌設定與路徑定義，這部分維持原樣即可) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 專案根目錄，假設此腳本在專案根目錄下
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

HIGH_CAPACITY_CONGESTION_FILE = os.path.join(DATA_DIR, 'high_capacity_congestion.csv')
WENHU_CONGESTION_FILE = os.path.join(DATA_DIR, 'wenhu_congestion.csv')

os.makedirs(DATA_DIR, exist_ok=True)

# 【關鍵點1】定義了統一的、乾淨的欄位標準
FINAL_COLUMNS = [
    'timestamp', 
    'station_id', 
    'line_direction_cid',
    'car1_congestion', 
    'car2_congestion', 
    'car3_congestion', 
    'car4_congestion', 
    'car5_congestion', 
    'car6_congestion'
]

# --- (load_data 和 save_data 函數維持原樣，它們的設計是正確的) ---
def load_data(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return df.reindex(columns=FINAL_COLUMNS, fill_value=0)
        except pd.errors.EmptyDataError:
            logger.warning(f"⚠️ CSV 檔案 '{file_path}' 為空，將建立新的 DataFrame。")
            return pd.DataFrame(columns=FINAL_COLUMNS)
        except Exception as e:
            logger.error(f"❌ 載入檔案 '{file_path}' 時發生錯誤: {e}", exc_info=True)
            return pd.DataFrame(columns=FINAL_COLUMNS)
    logger.info(f"📁 檔案 '{file_path}' 不存在，將建立新的 DataFrame。")
    return pd.DataFrame(columns=FINAL_COLUMNS)

def save_data(df: pd.DataFrame, file_path: str):
    # 【關鍵點2】儲存前，強制 DataFrame 符合 FINAL_COLUMNS 的結構
    df_to_save = df.reindex(columns=FINAL_COLUMNS, fill_value=0)
    df_to_save.to_csv(file_path, index=False, mode='w', header=True)
    logger.info(f"📊 已將 {len(df_to_save)} 筆資料儲存到 {file_path}")

# --- (process_* 函數維持原樣，它們的設計是正確的) ---

def process_high_capacity_data(raw_data: list[dict]) -> pd.DataFrame:
    processed_records = []
    for item in raw_data:
        try:
            # 【關鍵點3】從複雜的 API 回應中，只挑選我們需要的欄位，並對應到標準名稱
            record = {
                'timestamp': item.get('utime', ''),
                'station_id': item.get('StationID', ''),
                'line_direction_cid': int(item.get('CID', '0')) if str(item.get('CID', '0')).isdigit() else 0,
                'car1_congestion': item.get('Cart1L', '0'),
                'car2_congestion': item.get('Cart2L', '0'),
                'car3_congestion': item.get('Cart3L', '0'),
                'car4_congestion': item.get('Cart4L', '0'),
                'car5_congestion': item.get('Cart5L', '0'),
                'car6_congestion': item.get('Cart6L', '0')
            }
            if not all([record['timestamp'], record['station_id']]):
                logger.warning(f"高運量線記錄缺少必要欄位 (timestamp/station_id)，跳過: {item}")
                continue
            processed_records.append(record)
        except Exception as e:
            logger.error(f"❌ 處理高運量線單筆資料時發生錯誤: {item} - {e}", exc_info=True)
            continue
    df = pd.DataFrame(processed_records)
    # 再次確保欄位結構正確
    df = df.reindex(columns=FINAL_COLUMNS, fill_value=0)
    return df

def process_wenhu_data(raw_data: list[dict]) -> pd.DataFrame:
    processed_records = []
    for item in raw_data:
        try:
            # 【關鍵點4】同樣地，處理文湖線資料，並手動補上不存在的車廂
            record = {
                'timestamp': item.get('UpdateTime', ''),
                'station_id': item.get('StationID', ''),
                'line_direction_cid': int(item.get('CID', '0')) if str(item.get('CID', '0')).isdigit() else 0,
                'car1_congestion': item.get('Car1', '0'),
                'car2_congestion': item.get('Car2', '0'),
                'car3_congestion': item.get('Car3', '0'),
                'car4_congestion': item.get('Car4', '0'),
                'car5_congestion': '0', # 文湖線固定為0
                'car6_congestion': '0'  # 文湖線固定為0
            }
            if not all([record['timestamp'], record['station_id']]):
                logger.warning(f"文湖線記錄缺少必要欄位 (timestamp/station_id)，跳過: {item}")
                continue
            processed_records.append(record)
        except Exception as e:
            logger.error(f"❌ 處理文湖線單筆資料時發生錯誤: {item} - {e}", exc_info=True)
            continue
    df = pd.DataFrame(processed_records)
    df = df.reindex(columns=FINAL_COLUMNS, fill_value=0)
    return df

# --- (主邏輯 collect_and_save_congestion_data 維持原樣) ---
def collect_and_save_congestion_data():
    logger.info("--- [Collector] 開始新一輪資料收集 ---")
    
    # 高運量線
    logger.info("--- 嘗試獲取高運量線資料 ---")
    high_capacity_raw_data = metro_soap_api.get_high_capacity_car_weight_info()
    if high_capacity_raw_data:
        processed_df = process_high_capacity_data(high_capacity_raw_data)
        existing_df = load_data(HIGH_CAPACITY_CONGESTION_FILE)
        combined_df = pd.concat([existing_df, processed_df]).drop_duplicates(subset=['timestamp', 'station_id', 'line_direction_cid'], keep='last').reset_index(drop=True)
        save_data(combined_df, HIGH_CAPACITY_CONGESTION_FILE)
        logger.info(f"    -> ✅ 高運量線處理完畢。新獲取 {len(processed_df)} 筆，目前總共 {len(combined_df)} 筆。")
    else:
        logger.error("    -> ❌ 從 API 獲取高運量線原始資料失敗。")

    # 文湖線
    logger.info("--- 嘗試獲取文湖線資料 ---")
    wenhu_raw_data = metro_soap_api.get_wenhu_car_weight_info()
    if wenhu_raw_data:
        processed_df = process_wenhu_data(wenhu_raw_data)
        existing_df = load_data(WENHU_CONGESTION_FILE)
        combined_df = pd.concat([existing_df, processed_df]).drop_duplicates(subset=['timestamp', 'station_id', 'line_direction_cid'], keep='last').reset_index(drop=True)
        save_data(combined_df, WENHU_CONGESTION_FILE)
        logger.info(f"    -> ✅ 文湖線處理完畢。新獲取 {len(processed_df)} 筆，目前總共 {len(combined_df)} 筆。")
    else:
        logger.error("    -> ❌ 從 API 獲取文湖線原始資料失敗。")

    logger.info("--- [Collector] 資料收集完成，等待 5 分鐘後下一次更新... ---")


if __name__ == "__main__":
    while True:
        try:
            collect_and_save_congestion_data()
            time.sleep(5 * 60) # 等待 5 分鐘
        except KeyboardInterrupt:
            logger.info("--- [Collector] 收到手動中斷指令，程式正在關閉... ---")
            break
        except Exception as e:
            logger.critical(f"--- ❌ Collector 主迴圈發生嚴重錯誤: {e} ---", exc_info=True)
            logger.info("--- [Collector] 將在 5 分鐘後重試... ---\n")
            time.sleep(300)