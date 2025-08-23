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
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import dateparser
from dotenv import load_dotenv
from langchain_core.tools import tool

import config
from services import service_registry
from services.lost_item_search_service import lost_item_search_service
from services.realtime_mrt_service import RealtimeMRTService
from utils.exceptions import (DataLoadError, RouteNotFoundError,
                              StationNotFoundError)
from utils.station_name_normalizer import normalize_station_name


# ---------------------------------------------------------------------
# 2. 基本設定 (Basic Configuration)
# ---------------------------------------------------------------------
# 初始化日誌記錄器
logger = logging.getLogger(__name__)

# 載入環境變數 (若 .env 檔案存在)
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


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

# Emoji 對應表，用於美化擁擠度輸出
CONGESTION_EMOJI_MAP = {1: "😊 舒適", 2: "🤔 正常", 3: "😥 略多", 4: "😡 擁擠"}

# =====================================================================
# 4. Agent 工具定義 (Tool Definitions)
# =====================================================================

# ---------------------------------------------------------------------
# 1. 路徑規劃
# ---------------------------------------------------------------------#
@tool
def plan_route(start_station_name: str, end_station_name: str) -> str:
    """
    【路徑規劃專家】
    接收起點和終點站名，規劃路線並提供兩種資訊：
    1. 詳細、人性化的搭乘方向指引。
    2. 官方API提供的、包含所有停靠站的完整路徑列表。
    """
    logger.info(f"🚀 [路徑規劃] 開始規劃路徑：從「{start_station_name}」到「{end_station_name}」。")

    # 1. 驗證站名
    start_result = station_manager.get_station_ids(start_station_name)
    end_result = station_manager.get_station_ids(end_station_name)

    # ... (站名驗證邏輯與您提供的版本相同，此處為簡化省略，請保留您原有的驗證碼)
    if isinstance(start_result, dict) and 'suggestion' in start_result:
        return json.dumps({"error": "need_confirmation", **start_result}, ensure_ascii=False)
    if not start_result or not isinstance(start_result, list):
        return json.dumps({"error": f"抱歉，我找不到名為「{start_station_name}」的捷運站。"}, ensure_ascii=False)
    if isinstance(end_result, dict) and 'suggestion' in end_result:
        return json.dumps({"error": "need_confirmation", **end_result}, ensure_ascii=False)
    if not end_result or not isinstance(end_result, list):
        return json.dumps({"error": f"抱歉，我找不到名為「{end_station_name}」的捷運站。"}, ensure_ascii=False)

    start_sid = id_converter.tdx_to_sid(start_result[0])
    end_sid = id_converter.tdx_to_sid(end_result[0])

    # 2. 主要邏輯：優先使用官方API
    if start_sid and end_sid:
        logger.info(f"📞 嘗試呼叫北捷官方 SOAP API (SID: {start_sid} -> {end_sid})...")
        try:
            api_raw = metro_soap_api.get_recommended_route(start_sid, end_sid)
            
            if api_raw and isinstance(api_raw.get("path"), list) and len(api_raw["path"]) > 1:
                logger.info("✅ 成功從官方 API 獲取建議路線，開始進行雙重路徑處理...")
                
                # --- ✨ 核心改動 ✨ ---
                # 2.1 獲取原始的完整路徑列表
                full_path_list = api_raw["path"]
                
                # 2.2 產生人性化的搭乘指引
                detailed_directions = routing_manager.generate_directions_from_path(full_path_list)
                
                # 2.3 組合包含兩種資訊的最終訊息
                message = (
                    f"好的，從「{start_station_name}」到「{end_station_name}」的建議路線如下，預估時間約 {api_raw['time_min']} 分鐘：\n\n"
                    f"**搭乘指引：**\n" +
                    "\n".join(f"➡️ {step}" for step in detailed_directions) +
                    f"\n\n**行經車站：**\n" +
                    f"{' → '.join(full_path_list)}"
                )
                
                # 2.4 回傳包含所有資訊的 JSON
                return json.dumps({
                    "source": "official_api_enhanced",
                    "time_min": api_raw["time_min"],
                    "directions": detailed_directions, # 人性化指引
                    "full_path": full_path_list,       # 原始停靠站
                    "message": message
                }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"調用官方 SOAP API 或人性化處理時發生錯誤: {e}", exc_info=True)
    
    # 3. 備用方案 (保持不變，它本身就會回傳詳細資訊)
    logger.warning("SOAP API 無法使用或呼叫失敗，啟動備用方案：本地路網圖演算法。")
    try:
        fallback = routing_manager.find_shortest_path(start_station_name, end_station_name)
        if "path_details" in fallback:
            logger.info("✅ 成功透過本地演算法找到備用路徑。")
            fallback["message"] = "（備用方案）" + fallback["message"]
            return json.dumps({"source": "local_fallback", **fallback}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"本地路網規劃時發生未知錯誤: {e}", exc_info=True)

    # 4. 最終失敗
    logger.error(f"❌ 無法規劃路徑：從「{start_station_name}」到「{end_station_name}」，所有方法均失敗。")
    return json.dumps({"error": f"非常抱歉，我無法規劃從「{start_station_name}」到「{end_station_name}」的路線。"}, ensure_ascii=False)
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
def get_metro_line_info(line_name: str) -> str:
    """
    【捷運路網專家】
    當使用者詢問關於特定捷運「路線」的資訊時使用此工具。
    例如：「文湖線的起點和終點是哪裡？」、「板南線有哪些轉乘站？」
    它會回傳該路線的起訖站、所有車站和可轉乘的站點列表。
    """
    logger.info(f"🗺️ [路網查詢] 正在查詢路線資訊：{line_name}")
    
    # 標準化使用者可能輸入的簡稱
    normalized_map = {
        "棕": "文湖線", "文湖": "文湖線", "br": "文湖線",
        "紅": "淡水信義線", "淡水信義": "淡水信義線", "r": "淡水信義線",
        "綠": "松山新店線", "松山新店": "松山新店線", "g": "松山新店線",
        "橘": "中和新蘆線", "中和新蘆": "中和新蘆線", "o": "中和新蘆線",
        "藍": "板南線", "板南": "板南線", "bl": "板南線",
        "黃": "環狀線", "環狀": "環狀線", "y": "環狀線",
    }
    
    # 查找最符合的路線全名
    best_match_name = line_name
    for key, value in normalized_map.items():
        if key in line_name.lower():
            best_match_name = value
            break
            
    line_details = routing_manager.get_line_details(best_match_name)
    
    return json.dumps(line_details, ensure_ascii=False, indent=2)

# ✨✨✨ START: 新增的工具 ✨✨✨
@tool
def list_all_metro_lines() -> str:
    """
    【捷運路線盤點專家】
    當使用者詢問「有哪些捷運線？」或要求列出所有路線時使用此工具。
    它會回傳一個包含所有捷運線的名稱、代號和顏色的完整列表。
    """
    logger.info("🗺️ [路網查詢] 正在列出所有捷運路線...")
    all_lines = routing_manager.list_all_lines()
    return json.dumps(all_lines, ensure_ascii=False, indent=2)
# ✨✨✨ END: 新增的工具 ✨✨✨

@tool
def list_all_stations() -> str:
    """
    【捷運車站盤點專家】
    當使用者詢問「有哪些捷運站？」或要求列出所有車站時使用此工具。
    """
    logger.info("🚉 [車站查詢] 正在列出所有捷運車站...")
    station_names = station_manager.get_all_station_names()
    
    if not station_names:
        return json.dumps({"error": "無法獲取車站列表。"}, ensure_ascii=False)
        
    return json.dumps({
        "count": len(station_names),
        "stations": station_names,
        "message": f"台北捷運系統目前共有 {len(station_names)} 個車站。"
    }, ensure_ascii=False, indent=2)
# ✨✨✨ END: 新增的工具 ✨✨✨

@tool
def get_best_car_for_exit(station_name: str, direction: str, exit_number: int) -> str:
    """
    【最佳車廂推薦專家】
    當使用者到達某個捷運站，並想知道前往特定出口（例如3號出口）應該從哪個車廂下車最快時，使用此工具。
    你需要提供車站名稱、列車的行駛方向（終點站名稱），以及使用者想去的出口號碼。
    """
    logger.info(f" optimizing [最佳車廂推薦] 正在為「{station_name}」站，往「{direction}」方向，查詢靠近「{exit_number}」號出口的車廂。")

    # 1. 載入車廂出口對應資料
    car_exit_data = local_data_manager.car_exit_map
    if not car_exit_data:
        return json.dumps({"error": "車廂出口對應資料尚未載入。"}, ensure_ascii=False)

    # 2. 標準化站名以便搜尋
    norm_station = normalize_station_name(station_name)
    
    # 3. 尋找符合的車站、路線與方向
    found_cars = []
    station_info = None
    for item in car_exit_data:
        if normalize_station_name(item.get("station")) == norm_station:
            station_info = item
            break
            
    if not station_info:
        return json.dumps({"error": f"找不到「{station_name}」的車廂出口資料。"}, ensure_ascii=False)

    # 4. 尋找最匹配的方向 (處理 "往動物園" vs "動物園" 的情況)
    direction_data = None
    for dir_key, dir_value in station_info.get("Directions", {}).items():
        if direction in dir_key or dir_key in direction:
            direction_data = dir_value
            break
            
    if not direction_data:
         return json.dumps({"error": f"在「{station_name}」站找不到往「{direction}」方向的列車資訊。"}, ensure_ascii=False)

    # 5. 遍歷車廂列表，找出包含目標出口的車廂
    for car_info in direction_data.get("list", []):
        if exit_number in car_info.get("exits", []):
            found_cars.append(str(car_info.get("car")))

    # 6. 格式化回傳訊息
    if not found_cars:
        message = f"很抱歉，在「{station_name}」站往「{direction}」方向的列車，資料中沒有特別標示靠近 {exit_number} 號出口的車廂。建議您在月台留意出口指示圖。"
        return json.dumps({"station": station_name, "exit_number": exit_number, "found": False, "message": message}, ensure_ascii=False)
    
    car_str = "、".join(found_cars)
    message = f"好的！在「{station_name}」站下車後，若要前往 {exit_number} 號出口，建議您搭乘第 **{car_str}** 節車廂會最快抵達！"
    
    return json.dumps({
        "station": station_name,
        "direction": direction,
        "exit_number": exit_number,
        "recommended_cars": found_cars,
        "message": message
    }, ensure_ascii=False)

@tool
def get_realtime_mrt_info(station_name: str, destination: str) -> str:
    """
    【即時捷運到站專家】當使用者詢問「現在XX站往YY方向的車還有多久來」、「下一班車在哪裡」等關於
    特定車站和方向的即時列車資訊時，請使用此工具。這個工具會提供最即時的列車位置和到站倒數。

    Args:
        station_name (str): 使用者詢問的**目前**所在車站名稱。
        destination (str): 列車的行駛方向或終點站名稱。
    """
    logger.info(f"--- [工具(即時到站)] 查詢: {station_name} 往 {destination} 方向 ---")

    tool_output = {} # 初始化工具回傳的結構化數據

    try:
        current_query_time = datetime.now()

        realtime_mrt_service = service_registry.realtime_mrt_service
        station_manager = service_registry.station_manager # 確保取得 station_manager

        if not station_name or not destination:
            raise ValueError("請提供您所在的車站和列車的目的地。")

        # 解析並標準化使用者輸入的站名
        resolved_station_name = realtime_mrt_service.search_station(station_name)
        resolved_destination_name = realtime_mrt_service.search_station(destination)

        if not resolved_station_name:
            raise StationNotFoundError(f"我無法識別車站「{station_name}」。")
        if not resolved_destination_name:
            raise StationNotFoundError(f"我無法識別目的地「{destination}」。")

        # 獲取用於顯示給使用者的官方完整名稱
        official_station_display_name = station_manager.get_official_unnormalized_name(resolved_station_name)
        official_destination_display_name = station_manager.get_official_unnormalized_name(resolved_destination_name)

        # 推導出真正的列車終點站 (可能有多個，取第一個作為主要方向顯示)
        target_terminus_list = realtime_mrt_service.resolve_train_terminus(
            resolved_station_name, resolved_destination_name
        )

        if not target_terminus_list:
            tool_output = {
                "status": "No train found",
                "reason": "invalid_direction",
                "query_station": official_station_display_name,
                "query_destination": official_destination_display_name,
                "message_hint": f"從「{official_station_display_name}」站沒有往「{official_destination_display_name}」方向的直達列車。",
                "possible_directions": [station_manager.get_official_unnormalized_name(key) for key in station_manager.get_terminal_stations_for(resolved_station_name)]
            }
            return json.dumps(tool_output, ensure_ascii=False)


        candidate_trains = realtime_mrt_service.get_next_train_info(
            target_station_official_name=resolved_station_name,
            target_direction_normalized_list=target_terminus_list
        )

        if not candidate_trains:
            tool_output = {
                "status": "No train found",
                "reason": "no_realtime_data",
                "query_station": official_station_display_name,
                "query_destination": official_destination_display_name,
                "message_hint": f"目前沒有找到往「{official_destination_display_name}」方向的列車資訊。"
            }
        else:
            next_train_info = []
            for train in candidate_trains[:3]: # 只取最近的3班車
                countdown_str = train.get('CountDown', 'N/A')
                current_train_station = train.get('StationName', '未知車站')

                eta_seconds = None
                arrival_time_str = None
                
                if countdown_str == '列車進站':
                    eta_seconds = 0
                    arrival_time_str = (current_query_time).strftime('%H:%M') # 列車進站，視為立即到達
                else:
                    total_minutes = 0
                    # 嘗試解析 "X分鐘Y秒"
                    match_seconds = re.search(r'(\d+)\s*分鐘\s*(\d+)\s*秒', countdown_str)
                    # 嘗試解析 "X分鐘"
                    match_minutes = re.search(r'(\d+)\s*分鐘', countdown_str)
                    # 嘗試解析純數字 (例如： "5")
                    match_single_number = re.search(r'^(\d+)$', countdown_str.strip())

                    if match_seconds:
                        minutes = int(match_seconds.group(1))
                        seconds = int(match_seconds.group(2))
                        eta_seconds = minutes * 60 + seconds
                    elif match_minutes:
                        minutes = int(match_minutes.group(1))
                        eta_seconds = minutes * 60
                    elif match_single_number:
                        minutes = int(match_single_number.group(1))
                        eta_seconds = minutes * 60
                    
                    if eta_seconds is not None:
                        estimated_arrival_datetime = current_query_time + timedelta(seconds=eta_seconds)
                        arrival_time_str = estimated_arrival_datetime.strftime('%H:%M')
                    else:
                        # 如果無法解析，則使用原始倒數字串
                        countdown_str = countdown_str # 保持原始字串

                next_train_info.append({
                    "current_location": current_train_station,
                    "countdown_raw": countdown_str, # 原始倒數字串
                    "eta_seconds": eta_seconds, # 精確到秒的倒數
                    "arrival_time": arrival_time_str # 預計抵達的實際時間點 (HH:MM)
                })

            tool_output = {
                "status": "Success",
                "query_time": current_query_time.strftime('%H點%M分'),
                "query_station": official_station_display_name,
                "query_destination": official_destination_display_name,
                "train_terminus": station_manager.get_official_unnormalized_name(target_terminus_list[0]), # 確保是顯示名稱
                "next_trains": next_train_info,
                "suggestion": {
                    "text": "想知道這班車會不會很擠嗎？您可以問我「[車站名稱] 往 [目的地] 擠不擠」",
                    "example_query": f"{official_station_display_name} 往 {official_destination_display_name} 擠不擠"
                }
            }

        return json.dumps(tool_output, ensure_ascii=False)

    except StationNotFoundError as e:
        tool_output = {
            "status": "Error",
            "error_type": "Station Not Found",
            "message": f"😕 抱歉，我好像找不到您說的車站或目的地耶。錯誤訊息：{e}"
        }
        logger.warning(f"--- [工具(即時到站)] 查無車站或目的地: {e} ---")
        return json.dumps(tool_output, ensure_ascii=False)
    except ValueError as e:
        tool_output = {
            "status": "Error",
            "error_type": "Invalid Parameter/Direction",
            "message": f"🤔 哎呀，您提供的資訊好像有點問題，或是該方向沒有直達列車。錯誤訊息：{e}"
        }
        logger.warning(f"--- [工具(即時到站)] 參數錯誤或方向無效: {e} ---")
        return json.dumps(tool_output, ensure_ascii=False)
    except Exception as e:
        tool_output = {
            "status": "Error",
            "error_type": "Unknown Error",
            "message": "🤖 糟糕，我的捷運查詢系統好像出了一點小狀況，請稍後再試一次喔！"
        }
        logger.error(f"--- [工具(即時到站)] 發生未知錯誤: {e} ---", exc_info=True)
        return json.dumps(tool_output, ensure_ascii=False)



# 因為您真正的預測服務 CongestionPredictor (在 services/prediction_service.py) 本身就有更簡單、直接的方式來處理站名和方向，它並不需要這麼複雜的前置檢查。
@tool
def predict_train_congestion(station_name: str, direction: str, datetime_str: Optional[str] = None) -> str:
    """
    【捷運擁擠度預測專家】當使用者詢問「XX站擠不擠」、「YY站往ZZ方向人多嗎」這類關於車廂擁擠度的問題時，請使用此工具。
    它可以預測當前或未來特定時間的車廂擁擠程度。此工具常與 get_realtime_mrt_info 工具一起使用，來回答關於「車上人多不多」這類複合問題。

    Args:
        station_name (str): 預測的車站名稱。
        direction (str): 預測的行駛方向或終點站名稱。
        datetime_str (str, optional): 預測的日期和時間，可以是標準格式 `YYYY-MM-DD HH:MM`，
        也可以是自然語言表達，例如「明天早上八點」或「下一班車」。若未提供此參數，
        工具將自動使用當前時間進行預測。
    """
    logger.info(f"--- [工具(預測)] 原始查詢: {station_name} 往 {direction} 方向, 時間: {datetime_str} ---")

    if not station_name or not direction:
        return json.dumps({
            "error": "Missing parameters",
            "message": "🤔 哎呀，我需要知道您想查詢的「車站」和「方向」才能為您預測喔！"
        }, ensure_ascii=False)

    target_datetime = None
    if datetime_str:
        if datetime_str.lower() in ["現在", "即將", "馬上", "下一班車"]:
            target_datetime = datetime.now()
        else:
            target_datetime = dateparser.parse(
                datetime_str,
                settings={'PREFER_DATES_FROM': 'future', 'TIMEZONE': 'Asia/Taipei'}
            )
    
    if not target_datetime:
        target_datetime = datetime.now()
        logger.info("--- 未提供時間或無法解析，自動設定為當前時間 ---")

    now = datetime.now()
    if target_datetime > now + timedelta(days=365) or target_datetime < now - timedelta(days=1):
        logger.warning(f"--- ⚠️ 檢測到不合理的日期: {target_datetime.isoformat()}，可能為 LLM 幻覺。---")
        return json.dumps({
            "error": "Invalid time period",
            "message": f"📅 抱歉，您提供的日期 `{datetime_str}` 看起來有點太遙遠了。我只能預測一年內的擁擠度喔！今天的日期是 `{now.strftime('%Y-%m-%d')}`。"
        }, ensure_ascii=False)
        
    # --- 【核心修正】移除複雜且不存在的驗證，直接呼叫預測服務 ---
    prediction_result = congestion_predictor.predict_for_station(
        station_name=station_name, # 直接使用原始名稱
        direction=direction,     # 直接使用原始名稱
        target_datetime=target_datetime
    )

    if "error" in prediction_result:
        # 將服務層回傳的錯誤美化後再輸出
        error_msg = prediction_result['error']
        if "無法識別車站" in error_msg:
             return json.dumps({"message": f"😕 抱歉，我好像找不到「{station_name}」這個車站的資料耶。"}, ensure_ascii=False)
        return json.dumps({"message": f"😥 抱歉，預測時發生了一點小問題：{error_msg}"}, ensure_ascii=False)

    congestion_data = prediction_result.get("congestion_by_car", [])
    
    # 取得官方顯示名稱用於最終輸出
    official_station_display_name = prediction_result.get("station_name", station_name)
    official_direction_display_name = prediction_result.get("direction", direction)

    if congestion_data:
        time_display = target_datetime.strftime('%Y年%m月%d日 %H點%M分')
        if datetime_str and datetime_str.lower() in ["現在", "即將", "馬上", "下一班車"]:
            time_display = "現在"
                
        message_parts = [
            f"根據預測，在 {time_display} 往「{official_direction_display_name}」方向的列車擁擠度如下：",
            "---"
        ]
        
        for car in congestion_data:
            car_number = car['car_number']
            congestion_level = car['congestion_level']
            emoji_text = CONGESTION_EMOJI_MAP.get(congestion_level, "❔")
            message_parts.append(f"第 {car_number} 節車廂：{emoji_text}")
        
        max_congestion = max(c['congestion_level'] for c in congestion_data) if congestion_data else 0
        if max_congestion >= 3:
            message_parts.append("\n💡 **貼心提醒**：部分車廂可能人潮較多，建議您往較空曠的車廂移動喔！")
        elif max_congestion == 2:
            message_parts.append("\n😊 車廂狀況還不錯，人潮普通，可以輕鬆搭乘！")
        else:
            message_parts.append("\n🎉 太棒了！看起來車廂非常空曠，祝您有趟愉快的旅程！")
            
        final_message = "\n".join(message_parts)
    else:
        final_message = f"😥 抱歉，目前暫時無法取得「{official_station_display_name}」往「{official_direction_display_name}」方向在此時段的擁擠度預測資料。"

    response = {"message": final_message}
    return json.dumps(response, ensure_ascii=False)


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
    get_metro_line_info,
    list_all_metro_lines,
    list_all_stations,
]

logger.info(f"--- [Tools] 總共 {len(all_tools)} 個工具已成功註冊並準備就緒。 ---")