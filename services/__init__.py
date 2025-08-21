# services/__init__.py
"""
======================================================================
|               MetroPet AI Agent - Service Registry               |
|                  (融合 finyster & alan 分支功能)                   |
======================================================================
此檔案是整個應用程式的服務註冊中心 (Service Registry)。
它採用單例模式 (Singleton Pattern)，在程式啟動時，負責依序初始化
所有必要的服務，並提供一個統一的入口供其他模組獲取服務實例。
這樣可以確保所有服務只被建立一次，並由註冊中心集中管理。
"""

# ---------------------------------------------------------------------
# 1. 核心模組匯入 (Core Module Imports)
# ---------------------------------------------------------------------
import logging
import config
from utils.exceptions import ServiceInitializationError

# --- 匯入所有需要的服務類別和實例 ---
# Base services from both branches
from .fare_service import FareService
from .routing_service import RoutingManager
from .station_service import station_manager # 導入已實例化的 station_manager
from .local_data_service import local_data_manager # 導入已實例化的 local_data_manager
from .metro_soap_service import metro_soap_api # 導入已實例化的 metro_soap_api
from .tdx_service import tdx_api # 導入已實例化的 tdx_api

# finyster (HEAD) specific services
from .vector_search_service import vector_search_service
from .id_converter_service import id_converter_service
from .station_id_resolver import StationIdResolver

# alan/main specific services
from .prediction_service import CongestionPredictor
from .first_last_train_time_service import FirstLastTrainTimeService
from .realtime_mrt_service import RealtimeMRTService


# ---------------------------------------------------------------------
# 2. 基本設定 (Basic Configuration)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 3. ServiceRegistry 類別定義
# ---------------------------------------------------------------------
class ServiceRegistry:
    """
    一個集中管理所有業務服務實例的註冊中心 (Singleton)。
    """
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            logger.info("--- [ServiceRegistry] 正在建立新的服務註冊中心實例... ---")
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            if not cls._is_initialized:
                cls._instance._initialize_services()
                cls._is_initialized = True
        return cls._instance

    def _initialize_services(self):
        """
        在應用啟動時，依賴順序載入所有資料並初始化所有服務。
        """
        logger.info("--- [ServiceRegistry] 開始初始化所有服務... ---")
        try:
            # --- 步驟 1: 基礎服務與資料管理器 (通常無依賴) ---
            # 這些服務通常在模組載入時就已自行初始化
            self.local_data_manager = local_data_manager
            self.station_manager = station_manager
            self.tdx_api = tdx_api
            self.metro_soap_api = metro_soap_api
            
            # --- 步驟 2: 初始化 AI/輔助工具型服務 (依賴基礎服務) ---
            self.vector_search_service = vector_search_service
            self.id_converter_service = id_converter_service
            self.station_id_resolver = StationIdResolver(
                mapping_path=config.STATIONS_SID_MAP_PATH,
                main_station_info_path=config.STATION_DATA_PATH
            )
            self.congestion_predictor = CongestionPredictor(
                station_manager_instance=self.station_manager
            )

            # --- 步驟 3: 初始化核心業務邏輯服務 (依賴多個其他服務) ---
            self.fare_service = FareService(
                fare_data=self.local_data_manager.fares,
                station_id_map=self.station_manager.station_map
            )
            self.routing_manager = RoutingManager(
                station_manager_instance=self.station_manager,
                metro_soap_service_instance=self.metro_soap_api
            )
            self.first_last_train_time_service = FirstLastTrainTimeService(
                data_file_path=config.FIRST_LAST_TIMETABLE_DATA_PATH,
                station_manager=self.station_manager
            )
            self.realtime_mrt_service = RealtimeMRTService(
                metro_soap_api=self.metro_soap_api,
                station_manager=self.station_manager
            )
            # 啟動即時列車資訊的背景更新線程
            self.realtime_mrt_service.start_update_thread()

            logger.info("--- ✅ [ServiceRegistry] 所有服務已成功初始化並準備就緒。 ---")

        except Exception as e:
            logger.error(f"--- ❌ [ServiceRegistry] 服務初始化失敗: {e}", exc_info=True)
            raise ServiceInitializationError(f"核心服務初始化失敗: {e}")

    # --- Getter Methods: 提供外部模組獲取服務實例的統一入口 ---
    def get_fare_service(self) -> FareService: return self.fare_service
    def get_routing_manager(self) -> RoutingManager: return self.routing_manager
    def get_station_manager(self): return self.station_manager # 返回已實例化的模組
    def get_local_data_manager(self): return self.local_data_manager
    def get_tdx_api(self): return self.tdx_api
    def get_metro_soap_api(self): return self.metro_soap_api
    def get_congestion_predictor(self) -> CongestionPredictor: return self.congestion_predictor
    def get_first_last_train_time_service(self) -> FirstLastTrainTimeService: return self.first_last_train_time_service
    def get_realtime_mrt_service(self) -> RealtimeMRTService: return self.realtime_mrt_service
    def get_vector_search_service(self): return self.vector_search_service
    def get_id_converter_service(self): return self.id_converter_service
    def get_station_id_resolver(self) -> StationIdResolver: return self.station_id_resolver

# ---------------------------------------------------------------------
# 4. 全域單例 (Global Singleton)
# ---------------------------------------------------------------------
# 建立 ServiceRegistry 的一個全域實例，供整個應用程式使用。
service_registry = ServiceRegistry()