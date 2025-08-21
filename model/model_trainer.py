# model/model_trainer.py (專業分類模型升級版)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib
import os
import logging
from typing import Tuple, List
import json
import glob

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
    
    # 清理目標變數：確保擁擠度是 1, 2, 3, 4 其中之一
    df_melted['congestion'] = pd.to_numeric(df_melted['congestion'], errors='coerce')
    df_melted.dropna(subset=['congestion'], inplace=True)
    df_melted = df_melted[df_melted['congestion'].isin([1, 2, 3, 4])].astype({'congestion': int})

    # --- 【 ✨ 特徵工程 2.0 - 導入專家知識 ✨ 】 ---
    logger.info("      -> 正在創建 2.0 版特徵...")
    df_melted['timestamp'] = pd.to_datetime(df_melted['timestamp'])
    
    # (A) 更豐富的時間特徵
    df_melted['hour'] = df_melted['timestamp'].dt.hour
    # 【新增】更細粒度的時間特徵
    df_melted['minute'] = df_melted['timestamp'].dt.minute
    df_melted['day_of_week'] = df_melted['timestamp'].dt.dayofweek
    df_melted['is_weekend'] = (df_melted['day_of_week'] >= 5).astype(int)
    df_melted['is_peak_hour'] = df_melted['hour'].isin([7, 8, 17, 18, 19]).astype(int)

    # (B) 結合捷運路網的空間特徵 (Domain Knowledge)
    with open(os.path.join(DATA_DIR, 'mrt_station_info.json'), 'r', encoding='utf-8') as f:
        station_info = json.load(f)
    
    transfer_stations = {sid for info in station_info.values() if isinstance(info, dict) for sid in info.get('station_ids', []) if info.get('is_transfer')}
    df_melted['is_transfer_station'] = df_melted['station_id'].isin(transfer_stations).astype(int)
    
    # (C) 滯後特徵 (維持不變，但未來可強化)
    df_melted = df_melted.sort_values(by=['station_id', 'line_direction_cid', 'car_number', 'timestamp'])
    df_melted['lag_5min_congestion'] = df_melted.groupby(['station_id', 'line_direction_cid', 'car_number'])['congestion'].shift(1)
    df_melted['lag_1hr_congestion'] = df_melted.groupby(['station_id', 'line_direction_cid', 'car_number'])['congestion'].shift(12)
    df_melted.fillna(0, inplace=True)
    
    # 3. 類別特徵編碼 - 維持不變
    categorical_features = ['station_id', 'line_direction_cid']
    df_melted[categorical_features] = df_melted[categorical_features].astype(str)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(df_melted[categorical_features])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features), index=df_melted.index)
    
    # 4. 組合最終特徵
    numeric_features = [
        'hour', 'minute', 'day_of_week', 'is_weekend', 'is_peak_hour', 'is_transfer_station',
        'car_number', 'lag_5min_congestion', 'lag_1hr_congestion'
    ]
    
    final_df = pd.concat([df_melted[numeric_features].reset_index(drop=True), encoded_df.reset_index(drop=True), df_melted['congestion'].reset_index(drop=True)], axis=1)
    feature_columns = numeric_features + list(encoder.get_feature_names_out(categorical_features))
    
    # 【新增】特徵縮放
    scaler = StandardScaler()
    final_df[numeric_features] = scaler.fit_transform(final_df[numeric_features])
    
    logger.info(f"--- 預處理完成，共生成 {len(final_df)} 筆有效訓練樣本，使用 {len(feature_columns)} 個特徵。")
    return final_df, feature_columns, encoder, scaler

def train_and_save_model(df: pd.DataFrame, feature_columns: list, line_type: str, encoder: OneHotEncoder, scaler: StandardScaler):
    """
    【✨模型訓練與調優升級✨】
    使用 GridSearchCV 進行超參數調優，並使用分類評估指標。
    """
    logger.info(f"--- 開始訓練 {line_type} 分類模型... ---")
    
    X = df[feature_columns]
    y = df['congestion'] - 1 # 目標變數轉換為 0-indexed
    
    # 使用 stratify=y 確保訓練集和測試集中的各類別比例與原始數據相同
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 【新增】處理類別不平衡問題 (若需要)
    # 檢查 y_train 的類別分佈，如果某個類別樣本過少，可以啟用 SMOTE
    # logger.info(f"訓練集類別分佈: {np.bincount(y_train)}")
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    # logger.info(f"SMOTE 處理後訓練集類別分佈: {np.bincount(y_train)}")

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
        use_label_encoder=False,
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
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
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    logger.info(f"--- ✅ {line_type} 模型的 AUC-ROC: {auc_roc:.4f} ---")
    
    # 儲存產物
    output_dir = MODEL_DIR 
    best_model.save_model(os.path.join(output_dir, f'{line_type}_congestion_model.json'))
    joblib.dump(encoder, os.path.join(output_dir, f'{line_type}_encoder.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, f'{line_type}_scaler.joblib')) # 【新增】儲存 scaler
    pd.DataFrame(feature_columns, columns=['feature']).to_csv(os.path.join(output_dir, f'{line_type}_feature_columns.csv'), index=False)
    
    logger.info(f"      -> 模型相關產物已保存至: {output_dir}")

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
                logger.info(f"      -> 已刪除: {os.path.basename(file_path)}")
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