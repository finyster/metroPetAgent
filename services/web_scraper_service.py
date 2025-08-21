# services/web_scraper_service.py

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class WebScraperService:
    def __init__(self):
        # 台北捷運官網的車站資訊頁面
        self.base_url = "https://www.metro.taipei/cp.aspx?n=91974F2B13D997F1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_station_exit_info(self) -> dict:
        """
        從台北捷運官網爬取所有車站的出口詳細資訊。
        """
        logger.info("--- [WebScraper] 正在從台北捷運官網爬取最新的出口資訊... ---")
        try:
            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            all_exits_data = {}
            
            # 找到所有包含車站資訊的表格
            station_tables = soup.select("table.cp_table")
            
            for table in station_tables:
                # 找到表格標題中的車站 ID (例如 BL1、G2)
                caption = table.find("caption")
                if not caption: continue
                
                station_id_raw = caption.get_text(strip=True)
                # 清理格式，例如從 "(BL1) 板南線" 中提取 "BL1"
                station_id = station_id_raw.split(')')[0].replace('(', '').strip()

                if not station_id: continue

                exits_for_station = []
                # 遍歷表格中的每一行 (tr)，跳過標題行
                for row in table.find_all("tr")[1:]:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        exit_no = cells[0].get_text(strip=True)
                        description = cells[1].get_text(strip=True).replace('\n', ' ').replace('\r', ' ')
                        
                        # 過濾掉空的或無效的出口資訊
                        if exit_no and description and description != '無':
                             exits_for_station.append({
                                 "ExitNo": exit_no,
                                 "Description": description
                             })
                
                if exits_for_station:
                    all_exits_data[station_id] = exits_for_station
            
            logger.info(f"--- ✅ [WebScraper] 成功爬取並解析了 {len(all_exits_data)} 個站點的出口資訊。 ---")
            return all_exits_data

        except requests.RequestException as e:
            logger.error(f"--- ❌ [WebScraper] 爬取捷運出口資訊時發生網路錯誤: {e} ---", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"--- ❌ [WebScraper] 解析捷運出口資訊 HTML 時發生未知錯誤: {e} ---", exc_info=True)
            return {}

# 建立單一實例，方便在 build_database 中使用
web_scraper_service = WebScraperService()