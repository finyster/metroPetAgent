# agent/function_tools.py
"""
======================================================================
|                MetroPet AI Agent - Function Tools                |
|                  (融合 finyster & alan 分支功能)                   |
======================================================================
此檔案定義了所有可供 LangChain Agent 呼叫的工具函式。
整合了兩大分支的核心功能：
- finyster 分支: 專精於路徑規劃、生活資訊 (美食、設施、出口)、智慧搜尋。
- alan/main 分支: 專精於即時營運數據 (票價、班次、擁擠度、到站時間)。
"""

# ---------------------------------------------------------------------
# 1. 核心模組匯入 (Core Module Imports)
# ---------------------------------------------------------------------
import json
import logging
import random
import re, os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np

import dateparser
from dotenv import load_dotenv
from langchain_core.tools import tool

import config
from services import service_registry
from services.lost_item_search_service import lost_item_search_service
from services.realtime_mrt_service import RealtimeMRTService
from utils.exceptions import (DataLoadError, RouteNotFoundError,
                              StationNotFoundError)
from utils.time_parser import parse_countdown_to_seconds
from utils.station_name_normalizer import normalize_station_name

# ... (其他 import)
# ⚠️ 注意：我們需要一個新的 ManualSearchService，和 VectorSearchService 很像
# 為了簡化，我們先在這裡直接載入索引，未來您可以將其封裝成一個新服務
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq # 導入 ChatGroq




# ---------------------------------------------------------------------
# 2. 基本設定 (Basic Configuration)
# ---------------------------------------------------------------------
# 初始化日誌記錄器
logger = logging.getLogger(__name__)

# 載入環境變數 (若 .env 檔案存在)
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


# --- RAG 手冊搜尋設定 ---
manual_index_path = os.path.join(config.DATA_DIR, 'manual_vector.index')
manual_chunks_path = os.path.join(config.DATA_DIR, 'manual_chunks.json')
manual_search_is_ready = False
try:
    manual_retriever_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    manual_index = faiss.read_index(manual_index_path)
    with open(manual_chunks_path, 'r', encoding='utf-8') as f:
        import json
        manual_chunks = json.load(f)
    manual_search_is_ready = True
    logger.info("--- ✅ [RAG 手冊] 向量索引已成功載入。 ---")
except Exception as e:
    logger.warning(f"--- ⚠️ [RAG 手冊] 載入向量索引失敗: {e} ---")

# --- 新增一個專門用於「資訊提煉」的、便宜且快速的 LLM ---
distiller_llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    groq_api_key=config.GROQ_API_KEY
)
# --- RAG 設定結束 ---

# ---------------------------------------------------------------------
# 3. 服務實例化 (Service Instantiation)
# ---------------------------------------------------------------------
# 從 ServiceRegistry 獲取所有需要的服務單例，確保資源被集中管理且只初始化一次。
logger.info("--- [Tools] 正在從 ServiceRegistry 獲取所有服務實例... ---")
station_manager = service_registry.get_station_manager()
routing_manager = service_registry.get_routing_manager()
fare_service = service_registry.get_fare_service()
local_data_manager = service_registry.get_local_data_manager()
# 修正：alan/main 的 metro_soap_service 更完整，應使用 get_metro_soap_service
metro_soap_api = service_registry.get_metro_soap_api()
tdx_api = service_registry.get_tdx_api()
id_converter = service_registry.id_converter_service
congestion_predictor = service_registry.get_congestion_predictor()
first_last_train_time_service = service_registry.get_first_last_train_time_service()
realtime_mrt_service = service_registry.get_realtime_mrt_service()
# time_service = service_registry.get_time_service() # <- 舊的可以刪除或註解掉
llm_time_parser = service_registry.get_llm_time_parser_service() # ✨ 1. 獲取 LLMTimeParserService 實例

# ... (其他工具)


# Emoji 對應表，用於美化擁擠度輸出
CONGESTION_EMOJI_MAP = {
    1: "😊 空間舒適",
    2: "🤔 人潮普通",
    3: "😥 人潮略多",
    4: "😡 車廂擁擠"
}

# =====================================================================
# 4. Agent 工具定義 (Tool Definitions)
# =====================================================================

# ---------------------------------------------------------------------
# 1. 路徑規劃
# ---------------------------------------------------------------------#

@tool
def plan_route(start_station_name: str, end_station_name: str) -> str:
    """
    【路徑規劃專家 v3.3 - 邊界處理版】
    接收起點和終點站名，透過呼叫官方 API 智慧規劃所有可能的路線。
    此版本增加了對相同起終點的檢查，能提供更友善的回應。
    """
    logger.info(f"🚀 [路徑規劃 v3.3] 開始規劃路徑：從「{start_station_name}」到「{end_station_name}」。")

    # --- ✨✨✨【核心修正處：加入相同站點檢查】✨✨✨
    # 使用 normalize_station_name 來確保比較的一致性
    norm_start = normalize_station_name(start_station_name)
    norm_end = normalize_station_name(end_station_name)
    if norm_start == norm_end:
        return json.dumps({
            "error": "起點與終點站相同",
            "message": f"您已經在「{start_station_name}」囉！不需要再搭車了。😊"
        }, ensure_ascii=False)
    # --- ✨✨✨【修正結束】✨✨✨

    # 1. 驗證站名 (邏輯不變)
    start_result = station_manager.get_station_ids(start_station_name)
    end_result = station_manager.get_station_ids(end_station_name)

    if not start_result or not isinstance(start_result, list):
        return json.dumps({"error": f"抱歉，我找不到名為「{start_station_name}」的捷運站。"}, ensure_ascii=False)
    if not end_result or not isinstance(end_result, list):
        return json.dumps({"error": f"抱歉，我找不到名為「{end_station_name}」的捷運站。"}, ensure_ascii=False)

    # ... (後續的所有邏輯，包括重試和組合訊息，都保持不變)
    def get_line_name_from_scode(scode: str) -> str:
        line_map = {'BL': '板南線', 'BR': '文湖線', 'R': '淡水信義線', 'G': '松山新店線', 'O': '中和新蘆線', 'Y': '環狀線'}
        prefix = re.match(r"([A-Z]+)", scode)
        return line_map.get(prefix.group(1), "未知路線") if prefix else "未知路線"

    found_routes = {}
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1

    for start_tdx_id in start_result:
        for end_tdx_id in end_result:
            start_sid = id_converter.tdx_to_sid(start_tdx_id)
            end_sid = id_converter.tdx_to_sid(end_tdx_id)

            if not start_sid or not end_sid:
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    api_raw = metro_soap_api.get_recommended_route(start_sid, end_sid)
                    
                    is_valid_route = False
                    if api_raw and isinstance(api_raw.get("path"), list) and len(api_raw["path"]) > 1:
                        path_list = api_raw.get("path", [])
                        
                        if len(path_list) <= 2:
                            is_valid_route = True
                        else:
                            time_min = api_raw.get("time_min", 0)
                            if time_min > 1:
                                is_valid_route = True
                    
                    if is_valid_route:
                        path_key = tuple(api_raw["path"])
                        if path_key not in found_routes:
                            detailed_directions = routing_manager.generate_directions_from_path(api_raw["path"])
                            start_line_name = get_line_name_from_scode(start_tdx_id)
                            
                            found_routes[path_key] = {
                                "source": "official_api_enhanced",
                                "start_line": start_line_name,
                                "time_min": api_raw.get("time_min", 0),
                                "directions": detailed_directions,
                                "full_path": api_raw.get("path", [])
                            }
                        break 
                    
                    else:
                        logger.warning(f"--- ⚠️ API 回傳數據經驗證後不合理，正在進行第 {attempt + 1}/{MAX_RETRIES} 次重試... (SIDs: {start_sid} -> {end_sid}) ---")
                        time.sleep(RETRY_DELAY_SECONDS)

                except Exception as e:
                    logger.error(f"調用官方 SOAP API (SIDs: {start_sid} -> {end_sid}) 時發生錯誤: {e}", exc_info=True)
                    break
    
    if not found_routes:
        return json.dumps({"error": f"非常抱歉，我無法從官方 API 獲取到從「{start_station_name}」到「{end_station_name}」的有效路線資訊，請稍後再試。"}, ensure_ascii=False)

    final_routes = list(found_routes.values())
    
    if len(final_routes) == 1:
        route = final_routes[0]
        message = (
            f"好的，從「{start_station_name}」到「{end_station_name}」的建議路線如下，預估時間約 {route['time_min']} 分鐘：\n\n"
            f"**搭乘指引：**\n" +
            "\n".join(f"➡️ {step}" for step in route['directions']) +
            f"\n\n**行經車站：**\n" +
            f"{' → '.join(route['full_path'])}"
        )
        route["message"] = message
        return json.dumps(route, ensure_ascii=False)
    else:
        message_parts = [f"由於「{start_station_name}」或「{end_station_name}」是多條路線的交會站，為您找到以下幾種搭乘方案：\n"]
        for i, route in enumerate(final_routes):
            message_parts.append(f"\n--- **方案 {i+1}：從【{route['start_line']}】出發** ---")
            message_parts.append(f"預估時間約 {route['time_min']} 分鐘")
            message_parts.append("**搭乘指引：**")
            message_parts.append("\n".join(f"➡️ {step}" for step in route['directions']))
        
        final_message = "\n".join(message_parts)
        return json.dumps({
            "message": final_message,
            "routes": final_routes
        }, ensure_ascii=False)
    
# ---------------------------------------------------------------------
# 2. 票價查詢
# ---------------------------------------------------------------------
@tool
def get_mrt_fare(start_station_name: str, end_station_name: str) -> str:
    """
    【基礎票價查詢】當使用者僅詢問「多少錢」、「票價」、「費用」，但未指定特定身份（如老人、兒童、學生）時使用。
    此工具提供標準的「全票」和「兒童票」票價。
    如果使用者詢問特定票種（如愛心票、敬老票、學生票、台北市兒童票），請改用 `get_detailed_fare_info` 工具。
    """
    logger.info(f"--- [工具(基礎票價)] 查詢: {start_station_name} -> {end_station_name} ---")
    try:
        fare_info = fare_service.get_fare(start_station_name, end_station_name)
        message_parts = [f"從「{start_station_name}」到「{end_station_name}」的票價資訊如下："]
        
        if '全票' in fare_info:
            message_parts.append(f"全票為 NT${fare_info['全票']}。")
        if '兒童票' in fare_info:
            message_parts.append(f"兒童票為 NT${fare_info['兒童票']}。")
        
        if len(message_parts) == 1:
            message_parts.append("抱歉，目前沒有找到該路線的票價資訊。")
        else:
            message_parts.append("\n如需查詢愛心票、學生票等特殊票種，請提供您的乘客類型。")

        return json.dumps({
            "start_station": start_station_name,
            "end_station": end_station_name,
            "fare_details": fare_info,
            "message": "\n".join(message_parts)
        }, ensure_ascii=False)
    except StationNotFoundError as e:
        logger.warning(f"--- [工具(基礎票價)] 查詢時發生錯誤: {e} ---")
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"--- [工具(基礎票價)] 查詢時發生未知錯誤: {e} ---", exc_info=True)
        return json.dumps({"error": f"抱歉，查詢票價時發生內部問題。"}, ensure_ascii=False)
    
@tool
def get_detailed_fare_info(start_station_name: str, end_station_name: str, passenger_type: str) -> str:
    """
    【特殊票價專家】當使用者詢問特定身份或票種的票價時（例如「愛心票」、「敬老票」、「學生票」、「台北市兒童」、「新北市兒童」、「一日票」、「24小時票」），專門使用此工具。
    Args:
        start_station_name (str): 起點站名。
        end_station_name (str): 終點站名。
        passenger_type (str): 必須提供一個乘客類型，例如 "愛心票", "台北市兒童", "學生票", "一日票" 等。
    """
    logger.info(f"--- [工具(詳細票價)] 查詢: {start_station_name} -> {end_station_name}, 類型: {passenger_type} ---")
    try:
        fare_details = fare_service.get_fare_details(start_station_name, end_station_name, passenger_type)
        
        if "error" in fare_details:
            return json.dumps(fare_details, ensure_ascii=False)

        message = (
            f"從「{start_station_name}」到「{end_station_name}」，"
            f"「{passenger_type}」的票價為 NT${fare_details.get('fare', '未知')}。"
            f" ({fare_details.get('description', '無詳細說明')})"
        )
        
        fare_details["message"] = message
        return json.dumps(fare_details, ensure_ascii=False)
        
    except StationNotFoundError as e:
        logger.warning(f"--- [工具(詳細票價)] 查詢時發生錯誤: {e} ---")
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"--- [工具(詳細票價)] 查詢時發生未知錯誤: {e} ---", exc_info=True)
        return json.dumps({"error": f"抱歉，查詢詳細票價時發生內部問題。"}, ensure_ascii=False)

# ---------------------------------------------------------------------
# 3. 首末班車
# ---------------------------------------------------------------------
@tool
def get_first_last_train_time(station_name: str) -> str:
    """
    【暖心班次小助理】當使用者可能錯過列車，或是在深夜、清晨查詢班次時，用這個工具來查詢指定捷運站的首末班車時間。它會用友善貼心的方式回報，並提供溫馨提醒和可愛的小圖示。
    """
    logger.info(f"--- [工具(首末班車)] 查詢首末班車時間: {station_name} ---")
    
    first_last_train_time_service = service_registry.first_last_train_time_service

    if first_last_train_time_service is None:
        logger.error("FirstLastTrainTimeService 未初始化。請檢查 ServiceRegistry 的初始化流程。")
        return json.dumps({"error": "🥺 抱歉！目前捷運資訊服務好像有點小狀況，請您稍後再試試看喔！"}, ensure_ascii=False)

    try:
        timetable_data = first_last_train_time_service.get_timetable_for_station(station_name)
        
        if timetable_data:
            current_hour = datetime.now().hour

            # --- 訊息美化與個人化 ---

            # 隨機選擇開場白，增加變化性
            openings = [
                f"🎉 嗨嗨！我來幫您看看「{station_name}」站的班次喔！💪",
                f"💖 好的，馬上為您查詢「{station_name}」站的首末班車時間～ 請稍等一下下！",
                f"✨ 這是「{station_name}」站的詳細時刻表，希望對您有幫助喔！👇"
            ]
            message_parts = [random.choice(openings)]

            # 根據當前時間給予不同情境的提醒
            if current_hour >= 22 or current_hour <= 1:
                message_parts.append("\n🌙 現在時間比較晚囉，要特別注意末班車時間，別錯過囉！🏃‍♀️")
            elif 1 < current_hour <= 5:
                message_parts.append("\n😴 夜深了～您是不是正在等第一班車呢？我來幫您看看！☀️")
            else:
                message_parts.append("\n😊 這是您要查詢的固定班次資訊喔！")


            # 重新組織時刻表訊息，使其更清晰、更可愛
            for entry in timetable_data:
                destination = entry.get('destination_station', '未知終點站')
                first_train = entry.get('first_train_time', 'N/A')
                last_train = entry.get('last_train_time', 'N/A')
                service_days = entry.get('service_days', '每日行駛') # 加入 service_days 顯示

                # 簡化 service_days 顯示
                # 請注意：此處假定 service_days 的格式為 '{,1,1,1,1,1,1,1,1}' 代表每日
                # 如果您的實際數據有其他複雜的格式，可能需要更詳細的解析邏輯
                if service_days == "'{,1,1,1,1,1,1,1,1}'" or "1,1,1,1,1,1,1" in service_days: # 增加更寬鬆的判斷
                    service_days_display = "每日行駛"
                else:
                    service_days_display = "特定日行駛" # 如果有更複雜的服務日期，可能需要更詳細的解析

                line_info = (
                    f"\n➡️ 往 **{destination}** 方向：\n"
                    f"   ⏰ 首班車： **{first_train}**\n"
                    f"   ⏰ 末班車： **{last_train}**\n"
                    f"   🗓️ 營運日： {service_days_display}"
                )
                message_parts.append(line_info)

            # 隨機選擇結尾語
            closings = [
                "\n\n希望這個資訊對您有幫助，祝您旅途順利喔！🌈",
                "\n\n出門在外要注意安全，希望您能順利搭上車！💖",
                "\n\n如果時間有點趕，別忘了注意安全喔！有我在，您就安心搭車吧！�",
                "\n\n請您再確認一下時間，快樂出門，平安回家喔！😊"
            ]
            message_parts.append(random.choice(closings))

            # 保留官方的免責聲明，但用比較輕鬆的口吻
            message_parts.append("\n\n(✨ 貼心提醒：首末班車時間可能因維修、國定假日或特殊情況而變動，建議您提早一點到車站，並以車站現場公告為準最保險喔！)")

            # 使用兩個換行符號，讓最終呈現的訊息段落分明
            return json.dumps({
                "station": station_name, 
                "timetable": timetable_data, 
                "message": "\n".join(message_parts)
            }, ensure_ascii=False)
        
        # 查無資料的可愛回覆
        return json.dumps({"error": f"🧐 哎呀，好像沒有找到「{station_name}」站的首末班車資訊耶... \n這可能是因為該站目前沒有提供相關資料，或是資料正在更新中。\n您可以試著查詢其他車站，或是再確認一下站名是否有打錯喔！💡"}, ensure_ascii=False)
    
    except StationNotFoundError as e:
        logger.warning(f"--- [工具(首末班車)] 查詢時發生錯誤: {e} ---")
        # 找不到車站的可愛回覆
        return json.dumps({"error": f"😕 抱歉，我目前找不到「{station_name}」這個車站的資料耶。\n請確認您輸入的站名是不是正確的，或試試看其他相近的名稱喔！🗺️"}, ensure_ascii=False)
    except DataLoadError as e:
        logger.error(f"--- [工具(首末班車)] 數據載入錯誤: {e} ---", exc_info=True)
        # 資料載入失敗的可愛回覆
        return json.dumps({"error": "😴 抱歉，時刻表資料庫好像正在午休，現在無法查詢！請您稍後再試一次喔！⏰"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"--- [工具(首末班車)] 查詢時發生未知錯誤: {e} ---", exc_info=True)
        # 未知錯誤的可愛回覆
        return json.dumps({"error": f"🤖 糟糕，查詢「{station_name}」站的時候，發生了一點點小問題，技術人員正在努力搶修中！請您稍後再試試看喔！🛠️"}, ensure_ascii=False)

# ---------------------------------------------------------------------
# 4. 出口資訊
# ---------------------------------------------------------------------
@tool
def get_station_exit_info(station_name: str) -> str:
    """【車站出口專家】列出所有出口編號與描述。"""
    logger.info(f"[出口] {station_name}")
    station_ids = station_manager.get_station_ids(station_name)
    if not station_ids:
        return json.dumps({"error": f"找不到車站「{station_name}」。"}, ensure_ascii=False)

    exit_map = local_data_manager.exits
    exits: list[str] = []
    for sid in station_ids:
        exits.extend(
            f"出口 {e.get('ExitNo', 'N/A')}: {e.get('Description', '無描述')}"
            for e in exit_map.get(sid, [])
        )

    if not exits:
        return json.dumps({"error": f"查無「{station_name}」出口資訊"}, ensure_ascii=False)

    if all(x.endswith(": 無描述") for x in exits):
        msg = (f"「{station_name}」共有 {len(exits)} 個出入口，"
               "但暫無詳細描述。")
    else:
        msg = f"「{station_name}」出口資訊：\n" + "\n".join(exits)

    return json.dumps({"station": station_name, "exits": exits,
                       "message": msg}, ensure_ascii=False)

# ---------------------------------------------------------------------
# 5. 車站設施
# ---------------------------------------------------------------------
@tool
def get_station_facilities(station_name: str) -> str:
    """【車站設施專家】列出站內設施與描述。"""
    # 1. 記錄 Log，方便我們在後台看到 AI 何時呼叫了這個工具
    logger.info(f"[設施] {station_name}")

    # 2. 呼叫 StationManager，將使用者口語化的站名（如 "北車"）
    #    轉換成標準的車站 ID 列表（如 ["BL12", "R10"]）
    #    如果找不到，就直接回傳錯誤訊息。
    station_ids = station_manager.get_station_ids(station_name)
    if not station_ids:
        return json.dumps({"error": f"找不到車站「{station_name}」。"}, ensure_ascii=False)

    # 3. 讀取我們建立好的設施資料庫 (mrt_station_facilities.json)
    #    並根據上一步找到的車站 ID，把對應的詳細設施資訊撈出來。
    facilities = [
        local_data_manager.facilities.get(sid)
        for sid in station_ids
        if sid in local_data_manager.facilities
    ]
    # 移除可能的空值
    facilities = [f for f in facilities if f]

    # 4. 如果在資料庫中找不到任何資訊，回傳查無資料的錯誤。
    if not facilities:
        return json.dumps({"error": f"查無「{station_name}」設施資訊"}, ensure_ascii=False)

    # 5. 將找到的設施資訊（可能有多筆，針對轉乘站）合併成一個字串
    #    並建立一個友善的回覆訊息。
    desc = "\n".join(list(set(facilities))) # 使用 set 避免轉乘站資訊重複
    msg = f"「{station_name}」站的設施資訊如下：\n{desc}"

    # 6. 將最終結果包裝成 JSON 格式回傳給 AI Agent
    return json.dumps({
        "station": station_name, 
        "facilities_info": desc,
        "message": msg
    }, ensure_ascii=False)

# ---------------------------------------------------------------------
# 6. 遺失物智慧搜尋
# ---------------------------------------------------------------------
@tool
def search_lost_and_found(
    item_description: str | None = None, 
    station_name: str | None = None,
    date_str: str | None = None
) -> str:
    """
    【遺失物智慧搜尋專家】
    根據物品的模糊描述、可能的地點和日期（例如'昨天'或'2025/08/02'）來搜尋遺失物。
    """
    logger.info(f"[智慧遺失物搜尋] 正在搜尋: 物品='{item_description}', 車站='{station_name}', 日期='{date_str}'")
    
    if not item_description and not station_name:
        return json.dumps({"error": "缺少搜尋條件", "message": "請至少告訴我物品的描述或可能的車站喔！"}, ensure_ascii=False)

    # --- 【✨核心擴充✨】建立一個超級豐富的「物品別名地圖」 ---
    item_alias_map = {
        # ===== 電子票證類 =====
        "悠遊卡": "電子票證", "一卡通": "電子票證", "icash": "電子票證",
        "愛金卡": "電子票證", "ic卡": "電子票證", "學生卡": "電子票證",
        "敬老卡": "電子票證", "愛心卡": "電子票證",

        # ===== 3C / 電子產品類 =====
        "手機": "行動電話", "iphone": "行動電話",
        "airpods": "他類(耳機(無線)/藍牙)", "藍芽耳機": "他類(耳機(無線)/藍牙)", "無線耳機": "他類(耳機(無線)/藍牙)",
        "耳機": "他類(耳機",  # 使用不完整的詞，以匹配 "耳機)" 和 "耳機("
        "airpods": "他類(耳機(無線)/藍牙)",
        "airpods pro": "他類(耳機(無線)/藍牙)",
        "充電線": "他類(充電(傳輸)線)", "快充線": "他類(充電(傳輸)線)", "傳輸線": "他類(充電(傳輸)線)",
        "充電器": "他類(充電器)", "豆腐頭": "他類(充電器)",
        "行動電源": "他類(行動電源)", "充電寶": "他類(行動電源)",
        "電子菸": "他類(電子菸)",
        "相機": "照相機",

        # ===== 證件 / 卡片類 =====
        "身分證": "證件", "健保卡": "證件", "駕照": "證件", "學生證": "證件",
        "信用卡": "信用卡", "金融卡": "金融卡", "提款卡": "金融卡",
        "卡夾": "車票夾", "票卡夾": "車票夾",

        # ===== 雨具類 =====
        "雨傘": "傘", "陽傘": "傘",
        "折疊傘": "摺傘",
        "長柄傘": "長傘",

        # ===== 包包 / 袋子類 =====
        "錢包": "皮夾",
        "零錢袋": "零錢包",
        "提袋": "手提袋", "購物袋": "手提袋",
        "後背包": "背包", "書包": "背包",
        "塑膠袋": "塑膠袋",
        "紙袋": "紙袋",

        # ===== 衣物 / 飾品類 =====
        "衣服": "衣物", "外套": "衣物",
        "帽子": "帽子",
        "戒指": "戒指", "首飾": "首飾", "項鍊": "首飾", "手鍊": "首飾", "耳環": "耳環",
        "眼鏡": "眼鏡", "太陽眼鏡": "眼鏡",
        "手錶": "手錶",

        # ===== 其他常見「他類」物品 =====
        "筆": "他類(筆)", "原子筆": "他類(筆)",
        "手帕": "他類(手帕)",
        "束口袋": "他類(束口袋)",
        "吊飾": "他類(吊飾)", "鑰匙圈": "他類(吊飾)",

        # ===== 其他常見物品 =====
        "鑰匙": "鑰匙",
        "水壺": "水壺", "保溫瓶": "保溫瓶",
        "娃娃": "玩偶", "公仔": "玩偶",
    }
    # ----------------------------------------------------

    # --- 步驟 1: 處理日期 ---
    search_date = None
    if date_str:
        try:
            if "昨天" in date_str:
                search_date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
            elif "今天" in date_str:
                search_date = datetime.now().strftime('%Y/%m/%d')
            else:
                search_date = datetime.strptime(date_str, '%Y/%m/%d').strftime('%Y/%m/%d')
            logger.info(f"日期條件解析成功: {search_date}")
        except ValueError:
            logger.warning(f"無法解析日期字串: '{date_str}'，將忽略日期條件。")
            pass

    # --- 步驟 2: 處理地點 (站名 -> 站名 + 路線名) ---
    search_locations = set()
    if station_name:
        norm_station_name = station_name.replace("站", "").replace("駅", "")
        search_locations.add(norm_station_name)
        
        station_ids = station_manager.get_station_ids(station_name)
        if isinstance(station_ids, list) and station_ids:
            line_prefix_match = re.match(r"([A-Z]+)", station_ids[0])
            if line_prefix_match:
                line_prefix = line_prefix_match.group(1)
                line_map = {'BL': '板南線', 'BR': '文湖線', 'R': '淡水信義線', 'G': '松山新店線', 'O': '中和新蘆線', 'Y': '環狀線'}
                line_name = line_map.get(line_prefix)
                if line_name:
                    search_locations.add(line_name)
        logger.info(f"地點條件擴展為: {search_locations}")

    # --- 步驟 3: 處理物品 (精準別名 -> 語意搜尋) ---
    search_item_terms = set()
    if item_description:
        # 1. 優先使用「精準別名」進行轉換
        norm_item_desc = item_description.lower()
        if norm_item_desc in item_alias_map:
            official_item_name = item_alias_map[norm_item_desc]
            search_item_terms.add(official_item_name.lower())
            logger.info(f"物品 '{norm_item_desc}' 透過別名精準匹配到 '{official_item_name}'")
        
        # 2. 接著，使用「向量搜尋」來尋找其他語意相似的詞
        found_names = lost_item_search_service.find_similar_items(item_description, top_k=3, threshold=0.6)
        if found_names:
            search_item_terms.update(name.lower() for name in found_names)
            
        # 3. 無論如何，都將使用者原始的描述也加入搜尋目標
        search_item_terms.add(norm_item_desc)
        logger.info(f"物品條件擴展為: {search_item_terms}")

    # --- 步驟 4: 載入資料並執行最終篩選 ---
    try:
        with open(config.LOST_AND_FOUND_DATA_PATH, 'r', encoding='utf-8') as f:
            all_items = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.error(f"遺失物資料庫檔案遺失或損毀: {config.LOST_AND_FOUND_DATA_PATH}")
        return json.dumps({"error": "資料庫錯誤", "message": "抱歉，遺失物資料庫好像不見了。"}, ensure_ascii=False)

    filtered_items = all_items
    if search_date:
        filtered_items = [item for item in filtered_items if item.get('col_Date') == search_date]
    if search_locations:
        filtered_items = [item for item in filtered_items if any(loc.lower() in item.get('col_TRTCStation', '').lower() for loc in search_locations)]
    if search_item_terms:
        filtered_items = [item for item in filtered_items if any(term in item.get('col_LoseName', '').lower() for term in search_item_terms)]

    # --- 步驟 5: 格式化並回傳結果 ---
    top_results = filtered_items[:10]
    
    if not top_results:
        return json.dumps({"count": 0, "message": "很抱歉，在資料庫中找不到符合條件的遺失物。"}, ensure_ascii=False)

    formatted_results = [
        {"拾獲日期": item.get("col_Date"), "物品名稱": item.get("col_LoseName"), "拾獲車站": item.get("col_TRTCStation"), "保管單位": item.get("col_NowPlace")}
        for item in top_results
    ]
    
    return json.dumps({
        "count": len(top_results),
        "message": f"好的，幫您在資料庫中找到了 {len(top_results)} 筆最相關的遺失物資訊：",
        "results": formatted_results
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 7. 捷運美食搜尋
# ---------------------------------------------------------------------
@tool
def search_mrt_food(station_name: str, source_keyword: str | None = None) -> str:
    """
    【捷運美食家】
    根據使用者提供的捷運站名，查詢該站附近推薦的美食。
    可選擇性地根據來源關鍵字(例如 '米其林', '黃仁勳', '500碗')進行篩選。
    """
    # ✨ 新增：讓日誌也記錄下來源關鍵字，方便除錯
    logger.info(f"[美食搜尋] 正在搜尋「{station_name}」，來源關鍵字: '{source_keyword}'")

    # 1. 驗證並取得標準化的站名 (邏輯不變)
    station_ids = station_manager.get_station_ids(station_name)
    if not station_ids:
        return json.dumps({"error": f"找不到車站「{station_name}」。"}, ensure_ascii=False)

    # 2. 載入美食地圖資料 (邏輯不變)
    food_map = local_data_manager.food_map
    if not food_map:
        return json.dumps({"error": "美食地圖資料尚未載入。"}, ensure_ascii=False)
        
    # 3. 先找出該站點的「所有」餐廳 (邏輯不變)
    norm_station_name = normalize_station_name(station_name)
    all_restaurants_at_station = []
    for entry in food_map:
        if normalize_station_name(entry.get("station")) == norm_station_name:
            all_restaurants_at_station = entry.get("restaurants", [])
            break
    
    # ✨✨✨【核心修改】將篩選邏輯放在這裡 ✨✨✨
    # 檢查使用者是否提供了 `source_keyword`，並且我們確實找到了餐廳列表
    if source_keyword and all_restaurants_at_station:
        logger.info(f"--- 偵測到關鍵字 '{source_keyword}'，開始進行篩選...")
        
        filtered_restaurants = []
        # 遍歷每一家餐廳
        for restaurant in all_restaurants_at_station:
            # 取得餐廳的 source 欄位，可能是一個字串，也可能是一個列表
            source_info = restaurant.get("source", "")
            
            # 為了能統一處理，我們將 source 轉成一個 JSON 字串來進行比對
            # 這樣無論它是 "米其林" 還是 ["米其林", "500碗"]，都能被搜尋到
            source_text_for_search = json.dumps(source_info, ensure_ascii=False).lower()
            
            # 如果關鍵字存在於 source 的文字中，就將這家餐廳加入篩選結果
            if source_keyword.lower() in source_text_for_search:
                filtered_restaurants.append(restaurant)
        
        # 用篩選後的結果，覆蓋掉原本的餐廳列表
        found_restaurants = filtered_restaurants
        logger.info(f"--- 篩選完畢，找到 {len(found_restaurants)} 筆相符的結果。")
    else:
        # 如果沒有提供關鍵字，就使用全部的餐廳列表
        found_restaurants = all_restaurants_at_station

    # 4. 檢查最終是否有結果 (邏輯不變)
    if not found_restaurants:
        # 如果是篩選後沒有結果，可以給出更精確的提示
        if source_keyword:
             message = f"哎呀，在「{station_name}」附近，我找不到符合「{source_keyword}」這個來源的美食資訊耶。"
        else:
             message = f"哎呀，我目前還沒有收藏「{station_name}」附近的美食資訊耶。"
        
        return json.dumps({
            "station": station_name,
            "count": 0,
            "message": message
        }, ensure_ascii=False)

    # 5. 格式化並回傳最終結果 (邏輯不變)
    return json.dumps({
        "station": station_name,
        "count": len(found_restaurants),
        "message": f"好的，幫您找到了 {len(found_restaurants)} 家在「{station_name}」附近的美食：",
        "restaurants": found_restaurants
    }, ensure_ascii=False, indent=2)

@tool
def list_available_food_maps() -> str:
    """
    【美食地圖盤點專家】
    掃描美食資料庫，回傳所有不重複的美食地圖來源種類。
    """
    logger.info("[盤點資源] 正在掃描可用的美食地圖種類...")
    
    food_map = local_data_manager.food_map
    if not food_map:
        return json.dumps({"error": "美食地圖資料尚未載入。"}, ensure_ascii=False)

    unique_sources = set()
    for entry in food_map:
        for restaurant in entry.get("restaurants", []):
            source_info = restaurant.get("source")
            if not source_info:
                continue
            
            # 處理 source 是列表的情況 (例如: ["米其林", "500碗"])
            if isinstance(source_info, list):
                for s in source_info:
                    unique_sources.add(s)
            # 處理 source 是單一字串的情況
            elif isinstance(source_info, str):
                unique_sources.add(source_info)

    if not unique_sources:
        return json.dumps({"count": 0, "maps": []}, ensure_ascii=False)

    # 為了讓名稱更簡潔，可以做一些基本清理
    # 例如，從 "《台灣米其林指南2024》必比登推介地圖" 中取出 "必比登"
    cleaned_names = set()
    for s in unique_sources:
        if "必比登" in s:
            cleaned_names.add("米其林必比登推薦")
        elif "米其林" in s:
            cleaned_names.add("米其林星級餐廳")
        elif "黃仁勳" in s:
            cleaned_names.add("黃仁勳美食地圖")
        elif "500碗" in s:
            cleaned_names.add("500碗小吃地圖")
        elif "寵物友善" in s:
            cleaned_names.add("寵物友善餐廳")
        else:
            cleaned_names.add(s) # 如果沒有匹配，保留原名

    map_list = sorted(list(cleaned_names))

    return json.dumps({
        "count": len(map_list),
        "maps": map_list,
        "message": f"我這裡有 {len(map_list)} 種美食地圖可供參考：{', '.join(map_list)}。"
    }, ensure_ascii=False, indent=2)

@tool
def query_metro_network(
    query_type: str,
    line_name: Optional[str] = None
) -> str:
    """
    【捷運路網知識庫 v2.0 - 顏色增強版】
    處理所有關於捷運路網的知識型查詢。
    - query_type: 查詢類型，必須是 'list_lines' (所有路線), 'list_stations' (所有車站), 或 'line_details' (特定路線詳情)。
    - line_name: 當 query_type 為 'line_details' 時，必須提供要查詢的路線名稱。
    """
    logger.info(f"🗺️ [路網查詢 v2.0] 正在執行查詢，類型: {query_type}, 路線: {line_name}")

    # 建立一個從路線中文名反查路線代碼(SCODE)的字典，以便後續查找CSS class
    line_name_to_code_map = {
        '板南線': 'BL', '文湖線': 'BR', '淡水信義線': 'R',
        '松山新店線': 'G', '中和新蘆線': 'O', '環狀線': 'Y'
    }
    # 建立一個從路線代碼反查 CSS class 的字典
    code_to_class_map = {
        'BL': 'line-bl', 'BR': 'line-br', 'R': 'line-r',
        'G': 'line-g', 'O': 'line-o', 'Y': 'line-y'
    }

    if query_type == "list_lines":
        all_lines_data = routing_manager.list_all_lines()
        if "error" in all_lines_data:
            return json.dumps(all_lines_data, ensure_ascii=False)
        
        lines_summary = all_lines_data.get("lines", [])
        
        # ✨ 核心修改：為每一條路線加上顏色標籤
        message_parts = [f"台北捷運目前有 {len(lines_summary)} 條主要路線：\n"]
        for line_info in lines_summary:
            name = line_info.get("line_name", "")
            color = line_info.get("color", "")
            
            line_code = line_name_to_code_map.get(name, "")
            line_class = code_to_class_map.get(line_code, "")
            
            # 使用 Markdown 的列表格式，並在內部使用 HTML span 標籤
            colored_line_text = f'- <span class="{line_class}">{name} ({color})</span>'
            message_parts.append(colored_line_text)
            
        all_lines_data["message"] = "\n".join(message_parts)
        return json.dumps(all_lines_data, ensure_ascii=False, indent=2)
    
    elif query_type == "list_stations":
        # 這個查詢類型與路線顏色無關，保持不變
        station_names = station_manager.get_all_station_names()
        if not station_names:
            return json.dumps({"error": "無法獲取車站列表。"}, ensure_ascii=False)
        return json.dumps({
            "count": len(station_names),
            "stations": station_names,
            "message": f"台北捷運系統目前共有 {len(station_names)} 個車站。"
        }, ensure_ascii=False, indent=2)

    elif query_type == "line_details":
        if not line_name:
            return json.dumps({"error": "查詢路線詳情時，必須提供路線名稱。"}, ensure_ascii=False)
        
        # ... (標準化路線名稱的邏輯保持不變)
        normalized_map = {
            "棕": "文湖線", "文湖": "文湖線", "br": "文湖線",
            "紅": "淡水信義線", "淡水信義": "淡水信義線", "r": "淡水信義線",
            "綠": "松山新店線", "松山新店": "松山新店線", "g": "松山新店線",
            "橘": "中和新蘆線", "中和新蘆": "中和新蘆線", "o": "中和新蘆線",
            "藍": "板南線", "板南": "板南線", "bl": "板南線",
            "黃": "環狀線", "環狀": "環狀線", "y": "環狀線",
        }
        best_match_name = line_name
        for key, value in normalized_map.items():
            if key in line_name.lower():
                best_match_name = value
                break
        
        line_details_data = routing_manager.get_line_details(best_match_name)
        if "error" in line_details_data:
            return json.dumps(line_details_data, ensure_ascii=False)
            
        # ✨ 核心修改：為路線名稱加上顏色標籤
        name = line_details_data.get("line_name", "")
        color = line_details_data.get("color", "")
        stations = line_details_data.get("stations", [])
        
        line_code = line_name_to_code_map.get(name, "")
        line_class = code_to_class_map.get(line_code, "")

        colored_line_name = f'<span class="{line_class}">{name} ({color})</span>'
        
        line_details_data["message"] = f"{colored_line_name} 沿線車站包含：{'、'.join(stations)}。"
        return json.dumps(line_details_data, ensure_ascii=False, indent=2)

    else:
        return json.dumps({"error": f"不支援的查詢類型: {query_type}"}, ensure_ascii=False)

@tool
def get_best_car_for_exit(
    station_name: str, 
    exit_identifier: str, 
    start_station_name: Optional[str] = None
) -> str:
    """
    【下車站點優化專家 v2.0】
    當使用者想知道在某個車站下車後，前往特定出口應該搭乘哪節車廂時使用。
    - station_name: 抵達的車站名稱。
    - exit_identifier: 想去的出口編號或代碼 (例如 '2' 或 'M3')。
    - start_station_name (可選): 使用者的出發站。如果提供，可以回傳更精確的單一方向結果。
    """
    logger.info(f" optimizing [最佳車廂推薦 v2.0] 正在為「{station_name}」站查詢靠近「{exit_identifier}」的車廂，出發站：「{start_station_name}」。")

    car_exit_data = local_data_manager.car_exit_map
    if not car_exit_data:
        return json.dumps({"error": "車廂出口對應資料尚未載入。"}, ensure_ascii=False)

    norm_station = normalize_station_name(station_name)
    station_info = next((item for item in car_exit_data if normalize_station_name(item.get("station")) == norm_station), None)
            
    if not station_info:
        return json.dumps({"error": f"找不到「{station_name}」的車廂出口資料。"}, ensure_ascii=False)

    # ✨ 修正 bug：確保比對時，將出口編號都視為字串
    target_exit = str(exit_identifier).replace('號', '').replace('出口', '').strip()

    # 預先計算目標方向
    target_direction = None
    if start_station_name:
        # 使用 routing_manager 來解析出正確的終點站方向
        resolved_start = station_manager.resolve_station_alias(start_station_name)
        resolved_end = station_manager.resolve_station_alias(station_name)
        terminus_list = routing_manager.resolve_direction(resolved_start, resolved_end)
        if terminus_list:
            # 在 Directions 中，鍵值通常是 "往淡水", "往象山"
            target_direction = next((f"往{term}" for term in terminus_list if f"往{term}" in station_info.get("Directions", {})), None)
            logger.info(f"--- 透過出發站「{start_station_name}」，成功解析出目標方向為: {target_direction} ---")

    results_by_direction = {}
    directions_data = station_info.get("Directions", {})
    
    for direction_name, direction_details in directions_data.items():
        # 如果已算出目標方向，就只處理該方向的資料
        if target_direction and direction_name != target_direction:
            continue

        found_cars = []
        for car_info in direction_details.get("list", []):
            # ✨ 修正 bug：將資料中的出口也轉為字串進行比對
            if target_exit in [str(e) for e in car_info.get("exits", [])]:
                found_cars.append(str(car_info.get("car")))
        
        if found_cars:
            results_by_direction[direction_name] = found_cars

    if not results_by_direction:
        message = f"很抱歉，在「{station_name}」站的資料中，沒有找到靠近 {exit_identifier} 出口的車廂資訊。建議您在月台留意出口指示圖。"
        return json.dumps({"station": station_name, "exit_identifier": exit_identifier, "found": False, "message": message}, ensure_ascii=False)
    
    # 根據是否有 target_direction，決定回傳的訊息格式
    if target_direction and target_direction in results_by_direction:
        # 如果有明確方向，回傳簡潔的單一結果
        cars = results_by_direction[target_direction]
        car_str = "、".join(cars)
        message = f"好的！從「{start_station_name}」搭到「{station_name}」站後，若要前往 {exit_identifier} 出口，建議您搭乘第 **{car_str}** 節車廂會最快抵達！"
    else:
        # 如果沒有提供出發站，回傳所有可能的方向
        message_parts = [f"好的！如果您要在「{station_name}」站前往 {exit_identifier} 出口，建議的車廂位置如下：\n"]
        for direction, cars in results_by_direction.items():
            car_str = "、".join(cars)
            message_parts.append(f"**如果您搭乘的是開往【{direction.replace('往', '')}】的列車**，建議您搭乘第 **{car_str}** 節車廂。")
        message = "\n".join(message_parts)

    return json.dumps({
        "station": station_name,
        "exit_identifier": exit_identifier,
        "results": results_by_direction,
        "message": message
    }, ensure_ascii=False)

@tool
def get_realtime_mrt_info(start_station_name: str, end_station_name: str) -> str:
    """
    【智慧即時到站專家 v2.4 - 措辭精準版】
    當使用者詢問從 A 站到 B 站的「下一班車」時使用。
    此版本會用捷運官方的「終點站」來描述列車方向，提供最專業精準的回應。
    """
    logger.info(f"--- [智慧即時到站 v2.4] 查詢: 從「{start_station_name}」到「{end_station_name}」的下一班車 ---")

    # --- ✨✨✨【核心修改：建立輕量化的內部路徑查詢】✨✨✨
    def _get_route_for_direction_only(start_name: str, end_name: str) -> Optional[Dict]:
        """一個只為獲取方向而設計的輕量化內部路徑查詢器。"""
        start_ids = station_manager.get_station_ids(start_name)
        end_ids = station_manager.get_station_ids(end_name)
        if not start_ids or not end_ids: return None

        # 只需遍歷一次，找到第一條可用的路徑即可
        for start_tdx_id in start_ids:
            for end_tdx_id in end_ids:
                start_sid = id_converter.tdx_to_sid(start_tdx_id)
                end_sid = id_converter.tdx_to_sid(end_tdx_id)
                if not start_sid or not end_sid: continue
                
                try:
                    api_raw = metro_soap_api.get_recommended_route(start_sid, end_sid)
                    # 我們只關心路徑是否存在且長度大於1，完全忽略 time_min
                    if api_raw and isinstance(api_raw.get("path"), list) and len(api_raw["path"]) > 1:
                        directions = routing_manager.generate_directions_from_path(api_raw["path"])
                        return {"directions": directions} # 成功獲取，立即回傳
                except Exception as e:
                    logger.error(f"輕量化路徑查詢失敗 (SIDs: {start_sid} -> {end_sid}): {e}")
                    continue # 即使失敗也繼續嘗試下一個ID組合
        return None
    # --- ✨✨✨【修改結束】✨✨✨

    # --- ✨✨✨【核心邏輯修正】✨✨✨
    try:
        current_query_time = datetime.now()
        
        # 刪除舊的 plan_route.invoke() 呼叫，改為呼叫我們新建的輕量化查詢器
        route_data = _get_route_for_direction_only(start_station_name, end_station_name)
        
        # 檢查輕量化查詢器的結果
        if not route_data:
            raise Exception("內部輕量化路徑查詢失敗，找不到任何有效路徑。")

    except Exception as e:
        logger.error(f"在查詢即時到站時，內部路線規劃失敗: {e}", exc_info=True)
        return json.dumps({"error": "無法規劃路線以查詢即時資訊。"}, ensure_ascii=False)
    # --- ✨✨✨【修改結束】✨✨✨

    main_route = route_data.get("routes", [route_data])[0]
    first_direction_step = next((step for step in main_route.get('directions', []) if "方向" in step), None)
    if not first_direction_step: return json.dumps({"error": "無法從路線規劃中確定搭乘方向。"}, ensure_ascii=False)
    match = re.search(r"往「(.+?)」方向", first_direction_step)
    if not match: return json.dumps({"error": "無法從搭乘指引中解析出目的地。"}, ensure_ascii=False)
    direction_hint = match.group(1)

    try:
        resolved_start_name = station_manager.resolve_station_alias(start_station_name)
        terminus_list = routing_manager.resolve_direction(resolved_start_name, direction_hint)
        if not terminus_list: raise ValueError(f"無法將方向提示 '{direction_hint}' 解析為有效的終點站。")

        # --- ✨✨✨【核心修改：提取並使用正確的終點站名稱】✨✨✨
        # terminus_list[0] 儲存的就是我們解析出的、最精準的終點站名稱，例如 "南港展覽館站"
        actual_destination_name = terminus_list[0]
        # --- ✨✨✨【修改結束】✨✨✨

        next_trains_raw = realtime_mrt_service.get_next_train_info(
            target_station_official_name=resolved_start_name,
            target_direction_normalized_list=terminus_list
        )

        if not next_trains_raw:
            # ✨ 在提示訊息中也使用精準的終點站名稱
            return json.dumps({"message": f"目前查無從「{start_station_name}」往 **{actual_destination_name}** 方向的即時列車資訊，可能是末班車已駛離。"}, ensure_ascii=False)

        next_trains_processed = []
        for train in next_trains_raw:
            countdown_str = train.get('CountDown', '未知')
            arrival_time_str = None
            eta_seconds = parse_countdown_to_seconds(countdown_str)
            if eta_seconds != float('inf'):
                estimated_arrival_datetime = current_query_time + timedelta(seconds=eta_seconds)
                arrival_time_str = estimated_arrival_datetime.strftime('%H:%M')
            train['arrival_time'] = arrival_time_str
            next_trains_processed.append(train)

        first_train = next_trains_processed[0]
        countdown_text = first_train.get('CountDown', '未知')
        arrival_time = first_train.get('arrival_time')

        # ✨ 在回覆模板中使用精準的終點站名稱
        if "列車進站" in countdown_text:
            arrival_info = f"（{arrival_time}）" if arrival_time else ""
            message = f"快上車！從「{start_station_name}」搭乘往 **{actual_destination_name}** 方向的列車**正在進站**！{arrival_info} 🏃‍♀️"
        else:
            arrival_info = f"（大約 {arrival_time}）" if arrival_time else ""
            message = f"好的，從「{start_station_name}」搭乘往 **{actual_destination_name}** 方向的下一班列車，預計在 **{countdown_text}** 後抵達 {arrival_info}。"

        if len(next_trains_processed) > 1:
            second_train = next_trains_processed[1]
            second_countdown = second_train.get('CountDown', '未知')
            second_arrival = second_train.get('arrival_time')
            second_arrival_info = f"（大約 {second_arrival}）" if second_arrival else ""
            message += f"\n再下一班車約在 **{second_countdown}** 後抵達 {second_arrival_info}。"
        
        return json.dumps({
            "start_station": start_station_name,
            "end_station": end_station_name,
            "next_trains": next_trains_processed,
            "message": message
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"在獲取即時列車資訊時發生未知錯誤: {e}", exc_info=True)
        return json.dumps({"error": "查詢即時列車資訊時發生內部問題。"}, ensure_ascii=False)

# agent/function_tools.py

# ... (檔案頂部的 import)

@tool
def predict_train_congestion(
    start_station_name: str, 
    end_station_name: str, 
    datetime_str: Optional[str] = None
) -> str:
    """
    【智慧擁擠度預測專家 v3.2 - 輕量化路徑版】
    當使用者詢問從 A 站到 B 站的「車廂擁擠度」時使用。
    此版本在內部使用一個輕量化的路徑查詢，只獲取方向，忽略時間驗證，以提高穩定性。
    """
    logger.info(f"--- [工具(預測) v3.2] 查詢: 從「{start_station_name}」到「{end_station_name}」，時間: {datetime_str} ---")

    if not start_station_name or not end_station_name:
        return json.dumps({"message": "🤔 哎呀，我需要知道您的「起點」和「終點」才能為您預測喔！"}, ensure_ascii=False)
    
    # --- ✨ 2. 使用 LLM 驅動的時間解析服務 ---
    target_datetime = llm_time_parser.parse_datetime(datetime_str)
    # --- ✨ 簡化結束 ---

    # --- ✨✨✨【核心修改：插入輕量化的 plan_route 邏輯】✨✨✨
    def _get_route_for_direction_only(start_name: str, end_name: str) -> Optional[Dict]:
        """一個只為獲取方向而設計的輕量化內部路徑查詢器。"""
        start_ids = station_manager.get_station_ids(start_name)
        end_ids = station_manager.get_station_ids(end_name)
        if not (start_ids and isinstance(start_ids, list) and end_ids and isinstance(end_ids, list)): return None

        # 只需遍歷一次，找到第一條可用的路徑即可
        for start_tdx_id in start_ids:
            for end_tdx_id in end_ids:
                start_sid = id_converter.tdx_to_sid(start_tdx_id)
                end_sid = id_converter.tdx_to_sid(end_tdx_id)
                if not start_sid or not end_sid: continue
                
                try:
                    # 呼叫 API
                    api_raw = metro_soap_api.get_recommended_route(start_sid, end_sid)
                    
                    # 輕量化驗證：只要有路徑就接受，不管時間
                    if api_raw and isinstance(api_raw.get("path"), list) and len(api_raw["path"]) > 1:
                        directions = routing_manager.generate_directions_from_path(api_raw["path"])
                        return {"directions": directions} # 成功獲取，立即回傳
                except Exception as e:
                    logger.error(f"輕量化路徑查詢失敗 (SIDs: {start_sid} -> {end_sid}): {e}")
                    continue # 即使失敗也繼續嘗試下一個ID組合
        return None
    # --- ✨✨✨【修改結束】✨✨✨
        
    try:
        # 呼叫我們新建的輕量化查詢器
        route_data = _get_route_for_direction_only(start_station_name, end_station_name)
        if not route_data: raise Exception("內部輕量化路徑查詢失敗，找不到任何有效路徑。")
    except Exception as e:
        logger.error(f"在預測擁擠度時，內部路線規劃失敗: {e}", exc_info=True)
        return json.dumps({"error": "無法規劃路線以進行預測。"}, ensure_ascii=False)

    # ... (後續的程式碼，從解析方向到回傳訊息，都保持不變)
    first_direction_step = next((step for step in route_data.get('directions', []) if "方向" in step), None)
    if not first_direction_step: return json.dumps({"error": "無法從路線規劃中確定搭乘方向。"}, ensure_ascii=False)
    match = re.search(r"往「(.+?)」方向", first_direction_step)
    if not match: return json.dumps({"error": "無法從搭乘指引中解析出目的地。"}, ensure_ascii=False)
    
    direction_hint = match.group(1)

    try:
        resolved_start_name = station_manager.resolve_station_alias(start_station_name)
        terminus_list = routing_manager.resolve_direction(resolved_start_name, direction_hint)
        if not terminus_list: raise ValueError(f"無法將方向提示 '{direction_hint}' 解析為有效的終點站。")
        
        actual_terminal_name = terminus_list[0]
        
        prediction_result = congestion_predictor.predict_for_station(
            station_name=resolved_start_name,
            direction=actual_terminal_name,
            target_datetime=target_datetime
        )
    except Exception as e:
        logger.error(f"在執行預測時發生錯誤: {e}", exc_info=True)
        return json.dumps({"error": "執行擁擠度預測時發生內部問題。"}, ensure_ascii=False)

    # ... (組合最終回覆的程式碼不變)
    # 步驟 5: 組合最終回覆
    if "error" in prediction_result:
        return json.dumps({"message": f"😥 抱歉，預測時發生了一點小問題：{prediction_result['error']}"}, ensure_ascii=False)

    congestion_data = prediction_result.get("congestion_by_car", [])
    
    # --- ✨✨✨【核心修正】✨✨✨
    if congestion_data:
        time_display = "現在" if not datetime_str or datetime_str.lower() in ["現在", "即將", "馬上", "下一班車"] else target_datetime.strftime('%Y年%m月%d日 %H點%M分')
        
        message_parts = [
            f"好的，為您預測 {time_display} 從「{start_station_name}」出發往「{end_station_name}」方向的旅程：",
            f"您將搭乘往 **{actual_terminal_name}** 方向的列車，預計車廂擁擠度如下：",
            "---"
        ]
        
        for car in congestion_data:
            car_number = car['car_number']
            congestion_level = car['congestion_level']
            emoji_text = CONGESTION_EMOJI_MAP.get(congestion_level, "❔")
            message_parts.append(f"第 {car_number} 節車廂：{emoji_text}")
        
        max_congestion = max(c['congestion_level'] for c in congestion_data)
        if max_congestion >= 3:
            message_parts.append("\n💡 **貼心提醒**：部分車廂可能人潮較多！")
        else:
            message_parts.append("\n😊 車廂狀況還不錯，祝您旅途愉快！")
            
        # 將 final_message 的賦值移到 if 區塊的內部
        final_message = "\n".join(message_parts)
    else:
        final_message = f"😥 抱歉，目前暫時無法取得「{start_station_name}」往「{end_station_name}」方向在此時段的擁擠度預測資料。"
    # --- ✨✨✨【修正結束】✨✨✨

    return json.dumps({"message": final_message}, ensure_ascii=False)

@tool
def query_user_manual(user_question: str) -> str:
    """
    【RAG 知識問答專家 v2.2 - 最終微調版】
    當使用者提出關於如何使用此 AI 助理、有哪些功能，或任何開放式問題時使用此工具。
    它會根據使用者的問題，透過語意搜尋找到最相關的段落，並在相似度足夠高時提供精準的回答。
    """
    logger.info(f"📖 [RAG 手冊 v2.2] 正在根據問題搜尋知識庫: '{user_question}'")
    
    if not manual_search_is_ready:
        return json.dumps({"error": "抱歉，我的知識庫目前無法查閱，請稍後再試。"}, ensure_ascii=False)

    query_embedding = manual_retriever_model.encode([user_question])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = manual_index.search(query_embedding, k=1)
    
    # --- ✨✨✨【核心修正點：最終參數微調】✨✨✨
    # 將 L2 距離的門檻值從 1.2 提高到 1.25。
    # 這個微小的調整，就能讓 1.2170 這個優質的搜尋結果順利通過。
    DISTANCE_THRESHOLD = 1.25
    
    if len(indices[0]) > 0:
        best_distance = distances[0][0]
        logger.info(f"--- [RAG 手冊] 找到最相近的文件，距離為: {best_distance:.4f} (門檻為 < {DISTANCE_THRESHOLD}) ---")
        
        if best_distance < DISTANCE_THRESHOLD:
            best_match_index = indices[0][0]
            retrieved_chunk = manual_chunks[best_match_index]
            
            final_message = f"""
            根據我的使用者手冊，這裡有一段能回答「{user_question}」的相關資訊：
            
            ---
            {retrieved_chunk}
            ---
            
            希望這段說明對您有幫助！
            """

            return json.dumps({
                "user_question": user_question,
                "retrieved_context": retrieved_chunk,
                "message": final_message.strip()
            }, ensure_ascii=False, indent=2)

    logger.warning(f"--- [RAG 手冊] 所有找到的文件都未通過相關性門檻。 ---")
    return json.dumps({"message": "嗯...關於這個問題，我的手冊裡好像沒有提到耶。您可以換個方式問我嗎？"}, ensure_ascii=False)
    # --- ✨✨✨【修正結束】✨✨✨


# =====================================================================
# 最終工具列表 (Final Tool List)
# =====================================================================
# 匯集 finyster 和 alan 兩個分支的所有工具，打造功能全面的 Agent。
all_tools = [
    # 路徑與票務
    plan_route,
    get_mrt_fare,
    get_detailed_fare_info,
    # 即時營運
    get_first_last_train_time,
    get_realtime_mrt_info,
    predict_train_congestion,
    # 車站資訊
    get_station_exit_info,
    get_station_facilities,
    get_best_car_for_exit,
    # 生活與探索
    search_lost_and_found,
    search_mrt_food,
    list_available_food_maps,
    # 系統資訊
    query_metro_network,
    query_user_manual,
]

logger.info(f"--- [Tools] 總共 {len(all_tools)} 個工具已成功註冊並準備就緒。 ---")