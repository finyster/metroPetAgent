# services/routing_service.py

# ---------------------------------------------------------------------
# 核心模組匯入 (Core Module Imports)
# ---------------------------------------------------------------------
import json
import logging
import re
from typing import Any, Dict, List

import networkx as nx
import config
from services.metro_soap_service import MetroSoapService
from services.tdx_service import tdx_api
from utils.exceptions import RouteNotFoundError, StationNotFoundError
from utils.station_name_normalizer import normalize_station_name
from .station_service import StationManager

logger = logging.getLogger(__name__)

class RoutingManager:
    def __init__(self, station_manager_instance: StationManager, metro_soap_service_instance: MetroSoapService):
        logger.info("--- [Routing Service] 正在初始化並建立智慧捷運路網圖... ---")

        self.station_manager = station_manager_instance
        self.metro_soap_service = metro_soap_service_instance        
        self.station_id_to_name = self._load_station_id_map(self.station_manager.station_map)

        self.line_details: Dict[str, Dict[str, Any]] = {}

        self.graph = self._build_metro_graph()
        self.is_graph_ready = (self.graph is not None and self.graph.number_of_nodes() > 0)

        if self.is_graph_ready:
            logger.info("--- ✅ [Routing Service] 路網圖已成功初始化並準備就緒。 ---")
        else:
            logger.error("--- ❌ [Routing Service] 路網圖初始化失敗或為空。 ---")

    def _load_station_id_map(self, station_map: dict) -> dict:
        id_to_name = {}
        # 優先使用無英文/數字的作為官方中文名
        for name, ids in station_map.items():
            if not re.search('[a-zA-Z0-9]', name):
                for station_id in ids:
                    if station_id not in id_to_name:
                        id_to_name[station_id] = name.replace("臺", "台") + "站"
        
        # 補全可能因別名而遺漏的ID
        for name, ids in station_map.items():
            for station_id in ids:
                if station_id not in id_to_name:
                    # 使用標準化後的名稱作為備用
                    id_to_name[station_id] = name.replace("臺", "台") + "站"
        return id_to_name

    def _get_line_name_and_color(self, line_code: str) -> tuple[str, str, str]:
        # ✨ 升級：讓此函式同時回傳 CSS class 名稱
        line_map = {
            'BL': ('板南線', '藍線', 'line-bl'), 
            'BR': ('文湖線', '棕線', 'line-br'), 
            'R': ('淡水信義線', '紅線', 'line-r'),
            'G': ('松山新店線', '綠線', 'line-g'), 
            'O': ('中和新蘆線', '橘線', 'line-o'), 
            'Y': ('環狀線', '黃線', 'line-y')
        }
        return line_map.get(line_code, (line_code, '未知顏色', ''))

    def _build_metro_graph(self) -> nx.Graph:
        G = nx.Graph()
        all_routes_data = tdx_api.get_all_stations_of_route()
        if not all_routes_data:
            logger.error("--- ❌ [Routing] 無法從 TDX 獲取所有路線的站點資料，路網圖建立失敗。 ---")
            return G

        for route_info in all_routes_data:
            for station in route_info.get("Stations", []):
                station_id = station.get('StationID')
                if station_id and station_id not in G:
                    G.add_node(station_id, name=self.station_id_to_name.get(station_id, station_id))

        for route_info in all_routes_data:
            stations_on_this_route = route_info.get("Stations", [])
            if not stations_on_this_route:
                continue
            route_id = route_info.get('RouteID', '')
            line_code_match = re.match(r"([A-Z]+)", route_id)
            if not line_code_match:
                continue
            line_code = line_code_match.group(1)
            line_name, line_color, _ = self._get_line_name_and_color(line_code)

            if line_name not in self.line_details:
                terminus1_id = stations_on_this_route[0]['StationID']
                terminus2_id = stations_on_this_route[-1]['StationID']
                self.line_details[line_name] = {
                    "color": line_color,
                    "stations": [s['StationID'] for s in stations_on_this_route],
                    "terminus": [terminus1_id, terminus2_id]
                }

            for i in range(len(stations_on_this_route) - 1):
                u_id = stations_on_this_route[i]['StationID']
                v_id = stations_on_this_route[i+1]['StationID']
                if G.has_node(u_id) and G.has_node(v_id):
                     # 允許重複添加邊，但更新為ride類型和正確的line_name
                    G.add_edge(u_id, v_id, weight=3, type='ride', line_name=line_name)

        try:
            with open(config.TRANSFER_DATA_PATH, 'r', encoding='utf-8') as f:
                transfer_data = json.load(f)
            for transfer in transfer_data:
                u, v = transfer['FromStationID'], transfer['ToStationID']
                if G.has_node(u) and G.has_node(v):
                    G.add_edge(u, v, weight=5, type='transfer') # 轉乘權重較高
        except Exception as e:
            logger.warning(f"--- ⚠️ [Routing] 處理轉乘資訊時發生錯誤: {e} ---")

        logger.info(f"--- ✅ [Routing] 智慧捷運路網圖建立完成！共 {G.number_of_nodes()} 個站點，{G.number_of_edges()} 條連線。 ---")
        return G

    def _generate_directions_from_ids(self, path_ids: List[str]) -> List[str]:
        """
        【✨最終修正版 v3.1 - 顏色增強優化版✨】
        根據車站ID列表，透過偵測「路線變化」來生成帶有顏色的轉乘指引。
        此版本使用更穩健的方式來獲取路線顏色。
        """
        if not path_ids or len(path_ids) < 2:
            return ["路徑資訊不足，無法生成指引。"]

        # 建立一個從路線中文名反查路線代碼(SCODE)的字典，以便後續查找CSS class
        # 例如: {'板南線': 'BL', '淡水信義線': 'R', ...}
        line_name_to_code_map = {
            details['line_name']: code
            for code, details in {
                'BL': {'line_name': '板南線'}, 'BR': {'line_name': '文湖線'},
                'R': {'line_name': '淡水信義線'}, 'G': {'line_name': '松山新店線'},
                'O': {'line_name': '中和新蘆線'}, 'Y': {'line_name': '環狀線'}
            }.items()
        }

        directions = []
        start_node_name = self.station_id_to_name.get(path_ids[0], path_ids[0])
        directions.append(f"從「{start_node_name}」站上車。")

        current_line = None
        for i in range(len(path_ids) - 1):
            u_id, v_id = path_ids[i], path_ids[i+1]

            if not self.graph.has_edge(u_id, v_id): continue
            edge_data = self.graph.get_edge_data(u_id, v_id)

            if edge_data.get('type') == 'ride':
                line_name = edge_data.get('line_name')

                if line_name != current_line:
                    transfer_station_name = self.station_id_to_name.get(u_id, u_id)

                    # --- ✨✨✨【核心修改處】✨✨✨
                    # 移除了所有 HTML span 標籤的生成邏輯
                    line_color = self.line_details.get(line_name, {}).get("color", "")
                    plain_text_line_name = f"{line_name} ({line_color})"
                    # --- ✨✨✨【修改結束】✨✨✨

                    if current_line is not None:
                        directions.append(f"在「{transfer_station_name}」站，轉乘 {plain_text_line_name}。")

                    current_line = line_name

                    # 產生搭乘方向指引 (此部分邏輯不變)
                    line_info = self.line_details.get(current_line)
                    if line_info:
                        terminus1, terminus2 = line_info['terminus']
                        try:
                            dist_u_t1 = nx.shortest_path_length(self.graph, source=u_id, target=terminus1)
                            dist_v_t1 = nx.shortest_path_length(self.graph, source=v_id, target=terminus1)
                            direction_station_id = terminus1 if dist_v_t1 < dist_u_t1 else terminus2
                            direction_station_name = self.station_id_to_name.get(direction_station_id, direction_station_id)
                            directions.append(f"搭乘 {plain_text_line_name}，往「{direction_station_name}」方向。")
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            directions.append(f"搭乘 {plain_text_line_name}。")
                    else:
                        directions.append(f"搭乘 {plain_text_line_name}。")

        end_node_name = self.station_id_to_name.get(path_ids[-1], path_ids[-1])
        directions.append(f"在「{end_node_name}」站下車，抵達目的地。")

        return list(dict.fromkeys(directions)) # 使用 dict.fromkeys 移除重複語句，更高效

    # ... find_shortest_path 和 generate_directions_from_path 方法與上一版相同 ...
    # ... 為節省篇幅省略，請保留您檔案中這兩個方法的完整程式碼 ...
    def find_shortest_path(self, start_station_name: str, end_station_name: str) -> dict:
        if not self.is_graph_ready:
            raise RouteNotFoundError("抱歉，路網圖尚未準備好。")
        start_ids = self.station_manager.get_station_ids(start_station_name)
        end_ids = self.station_manager.get_station_ids(end_station_name)
        if not start_ids: raise StationNotFoundError(f"找不到起點站「{start_station_name}」。")
        if not end_ids: raise StationNotFoundError(f"找不到終點站「{end_station_name}」。")
        shortest_path_ids = None
        min_weight = float('inf')
        for s_id in start_ids:
            for e_id in end_ids:
                if self.graph.has_node(s_id) and self.graph.has_node(e_id):
                    try:
                        path_ids = nx.dijkstra_path(self.graph, source=s_id, target=e_id, weight='weight')
                        path_weight = nx.dijkstra_path_length(self.graph, source=s_id, target=e_id, weight='weight')
                        if path_weight < min_weight:
                            min_weight = path_weight
                            shortest_path_ids = path_ids
                    except nx.NetworkXNoPath:
                        continue
        if not shortest_path_ids:
            raise RouteNotFoundError(f"無法從「{start_station_name}」規劃到「{end_station_name}」。")
        final_path_description = self._generate_directions_from_ids(shortest_path_ids)
        estimated_time = round(min_weight)
        return {
            "start_station": start_station_name,
            "end_station": end_station_name,
            "path_details": final_path_description,
            "estimated_time_minutes": estimated_time,
            "message": f"從「{start_station_name}」到「{end_station_name}」的預估時間約為 {estimated_time} 分鐘。詳細路線：\n" + "\n".join(final_path_description)
        }

    def generate_directions_from_path(self, station_names: List[str]) -> List[str]:
        path_ids = []
        for name in station_names:
            norm_name = normalize_station_name(name)
            ids = self.station_manager.get_station_ids(norm_name)
            if ids and isinstance(ids, list):
                path_ids.append(ids[0])
            else:
                logger.warning(f"在從官方API路徑生成指引時，找不到站名 '{name}' (normalized: {norm_name}) 的ID。")
        
        if len(path_ids) < 2:
            logger.error(f"無法從官方路徑 '{station_names}' 解析出有效的ID路徑。")
            return ["抱歉，解析官方建議路線時發生錯誤。"]
            
        return self._generate_directions_from_ids(path_ids)
    
    def get_line_details(self, line_name: str) -> dict:
        """根據路線名稱，回傳該路線的詳細資訊。"""
        if line_name in self.line_details:
            details = self.line_details[line_name]
            station_names = [self.station_id_to_name.get(sid, sid) for sid in details['stations']]
            return {
                "line_name": line_name,
                "color": details['color'],
                "stations": station_names,
                "message": f"【{line_name} ({details['color']})】沿線車站包含：{'、'.join(station_names)}。"
            }
        return {"error": f"找不到名為「{line_name}」的路線資訊。"}

    def list_all_lines(self) -> dict:
        """回傳所有已知的捷運路線列表。"""
        if not self.line_details:
            return {"error": "目前沒有可用的路線資訊。"}
        
        lines_summary = [
            {"line_name": name, "color": details['color']}
            for name, details in self.line_details.items()
        ]
        
        return {
            "count": len(lines_summary),
            "lines": lines_summary,
            "message": f"台北捷運目前有 {len(lines_summary)} 條主要路線。"
        }

    def resolve_direction(self, start_station_name: str, direction_hint: str) -> List[str]:
        """
        【智慧方向解析 v2.0】
        根據起點站和方向提示（可以是終點站或任何一個中間站），解析出實際的列車終點站名稱。
        """
        start_ids = self.station_manager.get_station_ids(start_station_name)
        hint_ids = self.station_manager.get_station_ids(direction_hint)
        if not start_ids or not hint_ids:
            logger.warning(f"無法解析起點 '{start_station_name}' 或方向提示 '{direction_hint}' 的 ID。")
            return []

        possible_termini_ids = set()

        # 遍歷我們擁有的所有路線詳細資訊
        for line_name, details in self.line_details.items():
            line_stations = details.get('stations', []) # 該路線的所有車站 ID 列表
            
            # 檢查這條路線是否「同時」包含起點站和方向提示站
            start_id_on_line = next((sid for sid in start_ids if sid in line_stations), None)
            hint_id_on_line = next((hid for hid in hint_ids if hid in line_stations), None)

            # 如果兩個站都在同一條路線上
            if start_id_on_line and hint_id_on_line:
                start_pos = line_stations.index(start_id_on_line)
                hint_pos = line_stations.index(hint_id_on_line)

                # 如果索引不同（避免使用者輸入同一個站）
                if start_pos != hint_pos:
                    # 如果提示站的索引 > 起點站索引，代表列車是往路線列表的「末端」終點站開
                    if hint_pos > start_pos:
                        possible_termini_ids.add(details['terminus'][1]) # terminus 是一個 [起點ID, 終點ID] 的列表
                    # 反之，則是往列表的「開頭」終點站開
                    else:
                        possible_termini_ids.add(details['terminus'][0])
        
        if not possible_termini_ids:
            logger.warning(f"在任何單一路線上都找不到從 '{start_station_name}' 到 '{direction_hint}' 的有效方向。")
            return []

        # 將找到的終點站 ID 轉換回標準化後的站名
        final_termini_names = [self.station_manager.resolve_station_alias(self.station_id_to_name.get(tid, '')) for tid in possible_termini_ids]
        
        # 過濾掉空字串的結果並回傳
        return [name for name in final_termini_names if name]
    
    def get_terminal_stations_for(self, station_name: str) -> List[str]:
        """
        【新功能】根據提供的站名，找出該站所在的所有路線，並回傳這些路線的終點站名稱。
        """
        station_ids = self.station_manager.get_station_ids(station_name)
        if not station_ids:
            return []

        terminal_stations = set()
        for line_name, details in self.line_details.items():
            line_stations = details.get('stations', [])
            # 檢查這個站的任何一個 ID 是否存在於這條路線上
            if any(sid in line_stations for sid in station_ids):
                # 將這條路線的兩個終點站都加入到集合中
                for terminus_id in details.get('terminus', []):
                    terminus_name = self.station_id_to_name.get(terminus_id)
                    if terminus_name:
                        terminal_stations.add(terminus_name)

        return sorted(list(terminal_stations))
    
    def get_neighbor_stations(self, station_name: str) -> dict:
        """
        【新功能】根據站名，找出路網圖上所有直接相鄰的車站。
        """
        station_ids = self.station_manager.get_station_ids(station_name)
        if not station_ids or not isinstance(station_ids, list):
            return {"error": f"找不到車站「{station_name}」。"}

        neighbor_names = set()
        for sid in station_ids:
            if self.graph.has_node(sid):
                # graph.neighbors(sid) 會回傳所有與 sid 直接相連的節點
                for neighbor_id in self.graph.neighbors(sid):
                    # 我們只關心 'ride' 類型的鄰居，排除轉乘的內部節點
                    edge_data = self.graph.get_edge_data(sid, neighbor_id)
                    if edge_data and edge_data.get('type') == 'ride':
                        neighbor_name = self.station_id_to_name.get(neighbor_id, neighbor_id)
                        neighbor_names.add(neighbor_name)

        if not neighbor_names:
            return {"error": f"在路網圖中找不到「{station_name}」的鄰近車站。"}

        official_station_name = self.station_manager.resolve_station_alias(station_name)
        return {
            "station_name": official_station_name,
            "neighbors": sorted(list(neighbor_names))
        }
    
    def get_lines_for_station(self, station_name: str) -> dict:
        """
        【新功能】根據站名，找出該站點所屬的所有捷運路線。
        """
        # 使用 station_manager 解析站名，以處理別名並獲取所有可能的 ID (例如 "BL12", "R10")
        station_ids = self.station_manager.get_station_ids(station_name)
        if not station_ids or not isinstance(station_ids, list):
            return {"error": f"找不到車站「{station_name}」。"}

        found_lines = set()
        # 遍歷我們已知的每一條路線
        for line_name, details in self.line_details.items():
            line_stations_ids = details.get("stations", [])
            # 檢查這個站的任何一個 ID 是否存在於這條路線的站點列表中
            if any(sid in line_stations_ids for sid in station_ids):
                found_lines.add(line_name)
        
        if not found_lines:
            return {"error": f"在路網資料中找不到「{station_name}」所屬的任何捷運路線。"}

        # 整理找到的路線資訊，包含顏色等
        lines_with_details = []
        for line in sorted(list(found_lines)):
            details = self.line_details.get(line, {})
            # 從路線代碼反查 CSS class 以便前端美化
            line_code = ""
            if details.get("stations"):
                line_code_match = re.match(r"([A-Z]+)", details["stations"][0])
                if line_code_match:
                    line_code = line_code_match.group(1)
            
            _, line_color, line_class = self._get_line_name_and_color(line_code)
            lines_with_details.append({
                "line_name": line,
                "color": line_color,
                "line_class_css": line_class
            })
        
        official_station_name = self.station_manager.resolve_station_alias(station_name)
        return {
            "station_name": official_station_name,
            "lines": lines_with_details
        }
    
        