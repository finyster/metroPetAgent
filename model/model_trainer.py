# model/model_trainer.py (專業分類模型升級版)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE  # 啟用 SMOTE 處理不平衡
import numpy as np
import joblib
import os
import logging
from typing import Tuple, List
import json
import glob
from collections import Counter
import holidays  # 新增：導入 holidays 函式庫

# --- 配置日誌 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 路徑設置 ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def preprocess_for_training(filepath: str, line_type: str) -> Tuple[pd.DataFrame, List[str], OneHotEncoder, StandardScaler]:
    """
    【✨核心特徵工程升級 2.0✨】
    從原始 CSV 讀取資料，創建更豐富的時間與空間特徵，並為分類任務做準備。
    新增：引入節慶特徵、更多時間特徵，並優化滯後填充策略。
    """
    logger.info(f"--- 開始預處理 {line_type} 資料從 {filepath} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"資料檔案不存在: {filepath}。請先執行 data_collector.py。")
    
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"{filepath} 為空，無法進行訓練。")

    # 1. 資料格式轉換 (Wide to Long) - 維持不變
    num_cars = 4 if line_type == 'wenhu' else 6
    value_vars = [f'car{i}_congestion' for i in range(1, num_cars + 1)]
    id_vars = ['timestamp', 'station_id', 'line_direction_cid']
    
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='car_position', value_name='congestion')
    df_melted['car_number'] = df_melted['car_position'].str.extract(r'(\d+)').astype(int)
    
    df_melted['congestion'] = pd.to_numeric(df_melted['congestion'], errors='coerce')
    df_melted.dropna(subset=['congestion'], inplace=True)
    df_melted = df_melted[df_melted['congestion'].isin([1, 2, 3, 4])].astype({'congestion': int})

    # --- 【 ✨ 特徵工程 2.0 - 導入專家知識 ✨ 】 ---
    logger.info("        -> 正在創建 2.0 版特徵...")
    df_melted['timestamp'] = pd.to_datetime(df_melted['timestamp'], errors='coerce')
    original_rows = len(df_melted)
    df_melted.dropna(subset=['timestamp'], inplace=True)
    if len(df_melted) < original_rows:
        logger.warning(f"移除了 {original_rows - len(df_melted)} 筆因無效時間戳導致的資料。")
    
    # (A) 更豐富的時間特徵
    df_melted['hour'] = df_melted['timestamp'].dt.hour
    df_melted['minute'] = df_melted['timestamp'].dt.minute
    df_melted['day_of_week'] = df_melted['timestamp'].dt.dayofweek
    df_melted['month'] = df_melted['timestamp'].dt.month
    df_melted['year'] = df_melted['timestamp'].dt.year
    df_melted['is_weekend'] = (df_melted['day_of_week'] >= 5).astype(int)
    df_melted['is_morning_peak'] = ((df_melted['hour'] >= 7) & (df_melted['hour'] <= 9)).astype(int)
    df_melted['is_evening_peak'] = ((df_melted['hour'] >= 17) & (df_melted['hour'] <= 19)).astype(int)
    df_melted['is_peak_hour'] = ((df_melted['is_morning_peak'] == 1) | (df_melted['is_evening_peak'] == 1)).astype(int)

    logger.info("加入節慶特徵...")
    tw_holidays = holidays.TW()
    df_melted['is_holiday'] = df_melted['timestamp'].dt.date.apply(lambda x: x in tw_holidays).astype(int)

    # (B) 結合捷運路網的空間特徵
    with open(os.path.join(DATA_DIR, 'mrt_station_info.json'), 'r', encoding='utf-8') as f:
        station_info = json.load(f)
    
    transfer_stations = {sid for info in station_info.values() if isinstance(info, dict) for sid in info.get('station_ids', []) if info.get('is_transfer')}
    df_melted['is_transfer_station'] = df_melted['station_id'].isin(transfer_stations).astype(int)
    
    # (C) 滯後特徵
    df_melted = df_melted.sort_values(by=['station_id', 'line_direction_cid', 'car_number', 'timestamp'])
    
    df_melted['lag_5min_congestion'] = df_melted.groupby(['station_id', 'line_direction_cid', 'car_number'])['congestion'].shift(1)
    df_melted['lag_1hr_congestion'] = df_melted.groupby(['station_id', 'line_direction_cid', 'car_number'])['congestion'].shift(12)
    
    df_melted.loc[:, 'lag_5min_congestion'] = df_melted['lag_5min_congestion'].fillna(df_melted['lag_5min_congestion'].mean())
    df_melted.loc[:, 'lag_1hr_congestion'] = df_melted['lag_1hr_congestion'].fillna(df_melted['lag_1hr_congestion'].mean())
    
    # --- 關鍵修正：確保類別特徵處理的穩定性 ---
    categorical_features = ['station_id', 'line_direction_cid']
    
    # 這裡的 astype(str) 確保 OneHotEncoder 總是以字串形式處理這些欄位，與預測時一致。
    df_melted[categorical_features] = df_melted[categorical_features].astype(str)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(df_melted[categorical_features])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features), index=df_melted.index)
    
    # --- 關鍵修正：明確定義所有數值特徵 ---
    # `line_direction_cid` 被移除了，因為它現在是類別特徵
    numeric_features = [
        'hour', 'minute', 'day_of_week', 'month', 'year', 'is_weekend', 'is_morning_peak', 
        'is_evening_peak', 'is_peak_hour', 'is_holiday', 'is_transfer_station', 
        'car_number', 'lag_5min_congestion', 'lag_1hr_congestion'
    ]
    
    # 組合最終特徵
    final_df = pd.concat([df_melted[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True), df_melted['congestion'].reset_index(drop=True)], axis=1)
    
    # --- 關鍵修正：儲存完整的特徵名稱列表，包含獨熱編碼的欄位 ---
    feature_columns = numeric_features + list(encoder.get_feature_names_out(categorical_features))
    
    scaler = StandardScaler()
    final_df[numeric_features] = scaler.fit_transform(final_df[numeric_features])
    
    logger.info(f"--- 預處理完成，共生成 {len(final_df)} 筆有效訓練樣本，使用 {len(feature_columns)} 個特徵。")
    return final_df, feature_columns, encoder, scaler

def train_and_save_model(df: pd.DataFrame, feature_columns: list, line_type: str, encoder: OneHotEncoder, scaler: StandardScaler):
    """
    【✨模型訓練與調優升級✨】
    使用 GridSearchCV 進行超參數調優，並使用分類評估指標。
    新增：透過 SMOTE 過取樣處理類別不平衡問題。
    """
    logger.info(f"--- 開始訓練 {line_type} 分類模型... ---")
    
    X = df[feature_columns]
    y = df['congestion'] - 1  # 目標變數轉換為 0-indexed
    
    # 使用 stratify=y 確保訓練集和測試集中的各類別比例與原始數據相同
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 【 ✨ 處理類別不平衡問題 (SMOTE 過取樣) ✨ 】 ---
    # 計算原始訓練集類別分佈
    class_counts_original = Counter(y_train)
    logger.info(f"訓練集原始類別分佈: {class_counts_original}")

    # 新增：應用 SMOTE 進行少數類過取樣
    logger.info("應用 SMOTE 進行少數類過取樣以處理數據不平衡...")
    # 確保 k_neighbors 是有效的：它必須小於或等於最小類別中的樣本數量
    min_samples_in_minority_class = min(class_counts_original.values())
    smote_k_neighbors = min(5, min_samples_in_minority_class - 1)
    if smote_k_neighbors < 1: # 如果少數類樣本過少，SMOTE k_neighbors 將調整為 1
        logger.warning(f"少數類樣本數量 ({min_samples_in_minority_class}) 過少，SMOTE k_neighbors 將設定為 1。")
        smote_k_neighbors = 1

    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE 後的訓練集類別分佈: {Counter(y_train_res)}")

    # 【新增】使用 GridSearchCV 進行超參數調優
    logger.info("--- ⚙️ 開始使用 GridSearchCV 進行超參數調優... ---")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # 使用 TimeSeriesSplit 進行交叉驗證
    tscv = TimeSeriesSplit(n_splits=5)
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False,  # 停用舊版警告
        # 【修正】移除 enable_categorical=True，因為已使用 OneHotEncoder
        tree_method='hist', 
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1_weighted',  # 修改為加權 F1 以更好處理不平衡
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    # 【修正】移除 SMOTE 後的 sample_weights_res 傳遞給 fit 方法
    # 因為 SMOTE 已經平衡了數據，額外在 fit 時傳遞基於原始不平衡的權重是冗餘或可能產生反效果。
    grid_search.fit(X_train_res, y_train_res) 
    best_model = grid_search.best_estimator_
    
    logger.info(f"--- ✅ 超參數調優完成，找到最佳參數組合: {grid_search.best_params_} ---")

    # 使用最佳模型進行預測
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # --- 【 ✨ 核心修改：使用分類評估指標，並擴充 AUC-ROC ✨ 】 ---
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"--- ✅ {line_type} 模型訓練完成，評估 Accuracy (準確率): {accuracy:.4f} ---")
    
    # 打印更詳細的分類報告
    report = classification_report(y_test, y_pred, target_names=['舒適(1)', '正常(2)', '略多(3)', '擁擠(4)'])
    logger.info(f"\n--- 分類報告 ({line_type}) ---\n{report}")
    
    # 計算 AUC-ROC
    # 對於多分類，roc_auc_score 需要 multi_class='ovr' 或 'ovo'
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    logger.info(f"--- ✅ {line_type} 模型的 AUC-ROC: {auc_roc:.4f} ---")
    
    # 儲存產物
    output_dir = MODEL_DIR 
    best_model.save_model(os.path.join(output_dir, f'{line_type}_congestion_model.json'))
    joblib.dump(encoder, os.path.join(output_dir, f'{line_type}_encoder.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, f'{line_type}_scaler.joblib')) # 【新增】儲存 scaler
    pd.DataFrame(feature_columns, columns=['feature']).to_csv(os.path.join(output_dir, f'{line_type}_feature_columns.csv'), index=False)
    
    logger.info(f"        -> 模型相關產物已保存至: {output_dir}")

if __name__ == "__main__":
    logger.warning("--- 準備開始新一輪的『分類模型』訓練。已新增自動刪除舊模型檔案功能！ ---")
    
    # 【新增功能】自動刪除舊模型檔案
    # 尋找所有 line_type_congestion_model.json、_encoder.joblib、_feature_columns.csv、_scaler.joblib 檔案
    old_files = glob.glob(os.path.join(MODEL_DIR, '*_congestion_model.json')) + \
                glob.glob(os.path.join(MODEL_DIR, '*_encoder.joblib')) + \
                glob.glob(os.path.join(MODEL_DIR, '*_feature_columns.csv')) + \
                glob.glob(os.path.join(MODEL_DIR, '*_scaler.joblib'))
    
    if old_files:
        logger.info(f"--- 🗑️ 正在刪除 {len(old_files)} 個舊模型檔案... ---")
        for file_path in old_files:
            try:
                os.remove(file_path)
                logger.info(f"        -> 已刪除: {os.path.basename(file_path)}")
            except OSError as e:
                logger.error(f"刪除檔案 {file_path} 時發生錯誤: {e}")

    for line_type in ['high_capacity', 'wenhu']:
        filepath = os.path.join(DATA_DIR, f'{line_type}_congestion.csv')
        try:
            # 確保函數呼叫可以接收新增的回傳值
            processed_df, features, fitted_encoder, fitted_scaler = preprocess_for_training(filepath, line_type)
            # 確保函數呼叫可以傳遞新增的參數
            train_and_save_model(processed_df, features, line_type, fitted_encoder, fitted_scaler)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"--- ❌ {line_type} 訓練失敗: {e} ---")
            logger.error("請確保已運行 data_collector.py 並收集到足夠的資料。")
        except Exception as e:
            logger.critical(f"--- ❌ {line_type} 訓練過程中發生未知嚴重錯誤: {e} ---", exc_info=True)
    
    logger.info("\n--- 🎉 所有模型訓練流程結束！ ---")