# utils/exceptions.py
"""
======================================================================
|                MetroPet AI Agent - Custom Exceptions             |
|                  (融合 finyster & alan 分支功能)                   |
======================================================================
此檔案定義了應用程式中所有自定義的錯誤類別。
採用了 finyster 分支的基類繼承架構，並整合了 alan/main 分支中
所有更細緻的錯誤類型，以實現更精確的錯誤捕捉和處理。
"""

class MrtAgentBaseError(Exception):
    """應用程式所有自定義錯誤的基類。"""
    pass

# --- 核心業務邏輯錯誤 ---

class StationNotFoundError(MrtAgentBaseError):
    """當找不到指定的車站時引發。"""
    pass

class RouteNotFoundError(MrtAgentBaseError):
    """當找不到指定的路徑時引發。"""
    pass

# --- 系統與資料處理錯誤 ---

class ServiceInitializationError(MrtAgentBaseError):
    """當一個或多個核心服務初始化失敗時引發。"""
    pass

class DataLoadError(MrtAgentBaseError):
    """當從檔案或外部源載入資料失敗時引發。"""
    pass

class DataValidationError(MrtAgentBaseError):
    """當資料完整性或格式驗證失敗時引發。"""
    pass

# --- 外部與模型錯誤 ---

class ExternalAPIError(MrtAgentBaseError):
    """當呼叫外部 API 失敗或返回非預期響應時引發。"""
    pass

class PredictorError(MrtAgentBaseError):
    """當預測模型執行失敗或返回無效結果時引發。"""
    pass

class InvalidTimeFormatError(MrtAgentBaseError):
    """當時間格式無法被解析時引發。"""
    pass