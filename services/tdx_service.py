# services/tdx_service.py

import requests
import config
import time
import json 
import logging

logger = logging.getLogger(__name__)

class TDXApi:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://tdx.transportdata.tw/api/basic"
        self.auth_url = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
        self.access_token = None
        # 第一次獲取 Token
        self._get_access_token()

    def _get_access_token(self):
        """獲取 TDX Access Token"""
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials', 'client_id': self.client_id, 'client_secret': self.client_secret}
        try:
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=20)
            response.raise_for_status()
            logger.info("--- ✅ 成功獲取 TDX Access Token！ ---")
            self.access_token = response.json().get('access_token')
        except requests.RequestException as e:
            logger.error(f"--- ❌ 獲取 Access Token 失敗: {e} ---", exc_info=True)
            self.access_token = None
            
    def _get_api_data(self, url: str, retry: int = 5, delay: int = 10):
        """【強化版】API 資料獲取函式"""
        if not self.access_token:
            logger.error("--- ❌ 無法獲取 Access Token，無法進行 API 請求。 ---")
            return None
        
        headers = {'authorization': f'Bearer {self.access_token}', 'accept': 'application/json'}
        for attempt in range(retry):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 401:
                    logger.warning("--- ⚠️ Access Token 已過期或無效，正在重新獲取... ---")
                    self._get_access_token()
                    if not self.access_token: return None
                    headers['authorization'] = f'Bearer {self.access_token}'
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"--- ⚠️ 429 Too Many Requests，等待 {delay} 秒後重試 ({attempt + 1}/{retry}) ---")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"--- ❌ API 請求失敗 (HTTP Error) on URL: {url} ---", exc_info=False)
                    try:
                        logger.error(f"--- 錯誤詳情: {e.response.json()} ---")
                    except json.JSONDecodeError:
                        logger.error(f"--- 錯誤詳情 (非 JSON): {e.response.text} ---")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"--- ❌ API 請求發生嚴重錯誤 (RequestException) on URL: {url} ---", exc_info=True)
                return None
        
        logger.error(f"--- ❌ 在 {retry} 次重試後，依然無法從 URL 獲取資料: {url} ---")
        return None

    def _get_all_data_paginated(self, base_url: str, page_size: int = 500):
        """【強化版】分頁資料獲取函式"""
        all_data = []
        skip = 0
        while True:
            url_connector = "&" if "?" in base_url else "?"
            paginated_url = f"{base_url}{url_connector}$top={page_size}&$skip={skip}"
            
            page_data = self._get_api_data(paginated_url)
            
            if page_data is None:
                break
            
            if not page_data:
                break
            
            all_data.extend(page_data)
            
            if len(page_data) < page_size:
                break
            
            skip += page_size
            time.sleep(1.5) 

        return all_data if all_data else None

    # --- 以下所有 get_... 函式都不需修改，它們會自動繼承強化後的分頁能力 ---
    # --- ✨【核心新增處】✨ ---
    def get_all_stations(self):
        """獲取所有捷運站點的基本資料，這個 API 端點比 StationOfRoute 更完整。"""
        url = f"{self.base_url}/v2/Rail/Metro/Station/TRTC?$format=JSON"
        return self._get_all_data_paginated(url)

    # --- (其他 get_* 方法維持不變) ---
    def get_all_stations_of_route(self):
        url = f"{self.base_url}/v2/Rail/Metro/StationOfRoute/TRTC?$format=JSON"
        return self._get_all_data_paginated(url)

    def get_all_fares(self):
        url = f"{self.base_url}/v2/Rail/Metro/ODFare/TRTC?$format=JSON"
        return self._get_all_data_paginated(url, page_size=1000)

    def get_line_transfer_info(self):
        url = f"{self.base_url}/v2/Rail/Metro/LineTransfer/TRTC?$format=JSON"
        return self._get_all_data_paginated(url)

    def get_station_facilities(self):
        url = f"{self.base_url}/v2/Rail/Metro/StationFacility/TRTC?$format=JSON"
        return self._get_all_data_paginated(url)

    def get_station_exits(self, rail_system: str = "TRTC"):
        url = f"{self.base_url}/v2/Rail/Metro/StationExit/{rail_system}?$format=JSON"
        return self._get_all_data_paginated(url)
    
    def get_mrt_network(self):
        url = f"{self.base_url}/v2/Rail/Metro/Network/TRTC?$format=JSON"
        return self._get_all_data_paginated(url)

    def get_first_last_timetable(self, station_id: str):
        url = f"{self.base_url}/v2/Rail/Metro/FirstLastTimetable/TRTC?$filter=StationID eq '{station_id}'&$format=JSON"
        return self._get_api_data(url)
    
    def get_station_live_board(self, station_id: str) -> list[dict] | None:
        """
        【TDX API】【最終修正版】獲取指定捷運站的即時到站時刻表 (Live Board)。
        """
        # --- 【 ✨✨✨ 最終核心修正 ✨✨✨ 】 ---
        # 根據 TDX 官方文件，正確的端點是 /LiveBoard/TRTC，然後用 $filter 篩選 StationID
        # 這種方式同時適用於高運量和文湖線。
        api_url = f"{self.base_url}/v2/Rail/Metro/LiveBoard/TRTC"
        request_url = f"{api_url}?$filter=StationID eq '{station_id}'&$format=JSON&$orderby=EstimateTime"
        
        logger.info(f"--- [TDX] 正在從 {request_url} 獲取 {station_id} 的即時到站資訊... ---")
        response_data = self._get_api_data(request_url)

        if not response_data:
            logger.warning(f"--- [TDX] 未能獲取車站 {station_id} 的即時到站資訊。 ---")
            return None

        # 解析並格式化回應
        formatted_arrivals = []
        for board in response_data:
            try:
                arrival_minutes = round(board.get('EstimateTime', 0) / 60)
                # TDX API 有時會回傳 -1 代表末班車已離站，我們過濾掉這種情況
                if board.get('TripStatus') in [0, 1] and arrival_minutes >= 0:
                    formatted_arrivals.append({
                        # 使用 TripHeadSign 作為更準確的方向描述
                        "destination": board.get("TripHeadSign", "未知終點"),
                        "arrival_time_minutes": arrival_minutes
                    })
            except (TypeError, ValueError):
                continue

        logger.info(f"--- ✅ [TDX] 成功獲取並解析了 {len(formatted_arrivals)} 筆車站 {station_id} 的即時到站資訊。 ---")
        return formatted_arrivals

# 建立 TDXApi 的單一實例
tdx_api = TDXApi(client_id=config.TDX_CLIENT_ID, client_secret=config.TDX_CLIENT_SECRET)
