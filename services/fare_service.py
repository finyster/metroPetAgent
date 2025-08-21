from typing import Dict, Any, Optional # 修正：導入 Any 和 Optional 類型
from utils.exceptions import StationNotFoundError
from utils.station_name_normalizer import normalize_station_name # 導入標準化工具
import logging

logger = logging.getLogger(__name__)

class FareService:
    """
    負責處理所有與票價相關的業務邏輯。
    此服務在初始化時會載入票價資料，並提供查詢功能。
    """
    def __init__(self, fare_data: Dict[str, Dict[str, int]], station_id_map: Dict[str, list]):
        self._fare_data = fare_data
        self._station_id_map = station_id_map
        logger.info("FareService initialized.")

    def _get_station_ids_from_name(self, station_name: str) -> list[str]:
        """
        根據站名（或別名）取得所有對應的 station_id。
        """
        norm_name = normalize_station_name(station_name)
        if norm_name and norm_name in self._station_id_map:
            return self._station_id_map[norm_name]
        return []

    def get_fare(self, start_station_name: str, end_station_name: str) -> Dict[str, Any]:
        """
        根據使用者輸入的站名查詢基礎票價（全票和兒童票）。

        Args:
            start_station_name (str): 使用者輸入的起點站名。
            end_station_name (str): 使用者輸入的終點站名。

        Returns:
            Dict: 包含基礎票價的字典。

        Raises:
            StationNotFoundError: 如果任一站名無法識別或找不到票價。
        """
        logger.info(f"查詢基礎票價: {start_station_name} -> {end_station_name}")

        start_ids = self._get_station_ids_from_name(start_station_name)
        end_ids = self._get_station_ids_from_name(end_station_name)

        if not start_ids:
            raise StationNotFoundError(f"無法識別起點站名稱：'{start_station_name}'")
        if not end_ids:
            raise StationNotFoundError(f"無法識別終點站名稱：'{end_station_name}'")

        for s_id in start_ids:
            for e_id in end_ids:
                key1, key2 = f"{s_id}-{e_id}", f"{e_id}-{s_id}"
                if key1 in self._fare_data:
                    return self._fare_data[key1]
                if key2 in self._fare_data:
                    return self._fare_data[key2]
        
        raise StationNotFoundError(f"找不到從 '{start_station_name}' 到 '{end_station_name}' 的票價資訊。")

    def get_fare_details(self, start_station_name: str, end_station_name: str, passenger_type: str) -> Dict[str, Any]:
        """
        【新功能】根據乘客類型提供詳細或特殊票價資訊。
        目前此為模擬邏輯，未來可擴充以包含更複雜的票價規則。

        Args:
            start_station_name (str): 起點站名。
            end_station_name (str): 終點站名。
            passenger_type (str): 乘客類型 (例如 "愛心票", "台北市兒童", "新北市兒童", "學生票", "一日票")。

        Returns:
            Dict: 包含詳細票價資訊的字典。
        """
        logger.info(f"查詢詳細票價: {start_station_name} -> {end_station_name}, 乘客類型: {passenger_type}")

        # 先獲取基礎票價作為計算依據
        base_fare_info = self.get_fare(start_station_name, end_station_name)
        full_fare = base_fare_info.get("全票")

        if full_fare is None:
            return {"error": "無法獲取基礎票價，無法計算詳細票價。"}

        # --- 模擬票價計算邏輯 ---
        # 實際應用中，這裡的規則應來自更詳細的數據源或官方文件
        fare_rules = {
            "愛心票": {"discount": 0.4, "description": "依法令規定，享有半價優惠。"},
            "台北市兒童": {"discount": 0.6, "description": "設籍台北市之 6-12 歲兒童，享有 6 折優惠。"},
            "新北市兒童": {"discount": 0.4, "description": "設籍新北市之 6-12 歲兒童，享有 4 折優惠。"},
            "學生票": {"discount": 0.8, "description": "持有效學生證者，享有 8 折優惠（此為模擬）。"},
            "一日票": {"price": 150, "description": "當日營運時間內無限次搭乘。"},
            "24小時票": {"price": 180, "description": "首次進站後連續 24 小時內無限次搭乘。"}
        }

        rule = fare_rules.get(passenger_type)

        if not rule:
            return {"error": f"無法識別的乘客類型 '{passenger_type}'。"}

        if "price" in rule:
            calculated_fare = rule["price"]
        else:
            # 折扣票價以四捨五入計算到整數位
            calculated_fare = int(full_fare * rule.get("discount", 1.0) + 0.5)

        return {
            "passenger_type": passenger_type,
            "start_station": start_station_name,
            "end_station": end_station_name,
            "fare": calculated_fare,
            "description": rule.get("description", "無"),
            "base_full_fare": full_fare
        }