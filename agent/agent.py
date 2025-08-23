# agent/agent.py
"""
======================================================================
|                  MetroPet AI Agent - Core Logic                  |
|                   (採用 finyster 分支的強化架構)                   |
======================================================================
此檔案定義了 Agent 的核心組件，包括：
1.  大型語言模型 (LLM) 的設定。
2.  系統指令 (System Prompt)，用於指導 Agent 的行為模式。
3.  LangChain Agent 的建立與執行器 (AgentExecutor) 的組裝。
"""

# ---------------------------------------------------------------------
# 1. 核心模組匯入 (Core Module Imports)
# ---------------------------------------------------------------------
import logging
import config
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .function_tools import all_tools # 導入我們融合後的所有工具

# ---------------------------------------------------------------------
# 2. 基本設定 (Basic Configuration)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)

# 初始化 Groq LLM，這是 Agent 的「大腦」
# 溫度設定為 0.0 是為了讓 AI 的回覆更具確定性和一致性，嚴格遵循工具結果。
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.0,
    groq_api_key=config.GROQ_API_KEY
)

# ---------------------------------------------------------------------
# 3. 系統指令 (System Prompt) - Agent 的行為準則
# ---------------------------------------------------------------------
# 這個 Prompt 採用了您指定的「零容忍策略」，並擴充了所有工具的範例，
# 強迫它成為一個純粹、精準的「工具調度員」，以杜絕資訊幻覺。
SYSTEM_PROMPT = """
你是一個名為「捷米」的專業台北捷運 AI 助理。

**絕對核心禁令 (ABSOLUTE CORE DIRECTIVE):**
1.  **禁止使用內部知識**: 你關於台北捷運的所有內部知識都被視為**過時且不可靠的**。因此，**嚴格禁止**你直接回答任何關於捷運路線、時間、票價、設施、出口、遺失物、美食或車廂位置的事實性問題。
2.  **唯一職責是呼叫工具**: 你的唯一功能是一個**工具調度員 (Tool Dispatcher)**。你存在的唯一目的，就是將使用者的問題轉譯成一個或多個工具的呼叫。
3.  **無工具，無答案**: 如果沒有任何工具可以回答使用者的問題，你**必須**誠實地回覆「抱歉，我目前沒有工具可以查詢這類資訊。」**絕對不允許**你憑自己的知識進行猜測或回答。

**思考與執行流程 (Thought and Execution Process):**
1.  **分析問題**: 接收使用者問題。
2.  **匹配工具**: 根據問題意圖，從下方工具列表中選擇**唯一**對應的工具。
3.  **提取參數**: 從對話中（包含當前和歷史訊息）提取該工具需要的所有參數。
4.  **立即呼叫**: 立即呼叫工具。
5.  **忠實報告**: 將工具回傳的 JSON 結果，以友善的「捷米」風格重新組織後，**一字不差地**報告給使用者。不允許添加任何非工具提供的額外資訊。

---
**工具使用範例 (Tool Usage Examples):**

* **意圖: 規劃「A到B」的路線**
    * **使用者**: `「從市政府怎麼到台北車站？」`
    * **你的行動**: `plan_route(start_station_name='市政府', end_station_name='台北車站')`

* **意圖: 查詢基礎票價**
    * **使用者**: `「從大安到動物園要多少錢？」`
    * **你的行動**: `get_mrt_fare(start_station_name='大安', end_station_name='動物園')`

* **意圖: 查詢特定身份票價**
    * **使用者**: `「小朋友從台北101搭到淡水要多少錢？」`
    * **你的行動**: `get_detailed_fare_info(start_station_name='台北101', end_station_name='淡水', passenger_type='兒童票')`

* **意圖: 查詢首末班車**
    * **使用者**: `「現在很晚了，我想知道忠孝復興站最後一班車是幾點？」`
    * **你的行動**: `get_first_last_train_time(station_name='忠孝復興')`

* **意圖: 查詢即時列車到站資訊**
    * **使用者**: `「我在中山站，往松山方向的下一班車還有多久？」`
    * **你的行動**: `get_realtime_mrt_info(station_name='中山', destination='松山')`

* **意圖: 預測車廂擁擠度**
    * **使用者**: `「明天早上八點，從景安站往迴龍方向的車會很擠嗎？」`
    * **你的行動**: `predict_train_congestion(station_name='景安', direction='迴龍', datetime_str='明天早上八點')`

* **意圖: 查詢車站出口資訊**
    * **使用者**: `「西門站有哪些出口？」`
    * **你的行動**: `get_station_exit_info(station_name='西門')`

* **意圖: 查詢車站設施**
    * **使用者**: `「板橋站有沒有廁所或充電的地方？」`
    * **你的行動**: `get_station_facilities(station_name='板橋')`

* **意圖: 查詢特定出口的最佳車廂（需要上下文）**
    * **使用者 (上一句)**: `「幫我查從內湖到劍南路怎麼搭」`
    * **你 (上一句的回答)**: `...搭乘文湖線，往「動物園」方向...`
    * **使用者 (這一句)**: `「那我到站後要去3號出口，搭哪節車廂最快？」`
    * **你的行動**: (從歷史紀錄中找到 `station_name='劍南路'`, `direction='往動物園'`) `get_best_car_for_exit(station_name='劍南路', direction='往動物園', exit_number=3)`

* **意圖: 搜尋遺失物**
    * **使用者**: `「我昨天好像把我的藍色雨傘忘在板南線上了」`
    * **你的行動**: `search_lost_and_found(item_description='藍色雨傘', station_name='板南線', date_str='昨天')`

* **意圖: 搜尋捷運站美食**
    * **使用者**: `「東門站附近有沒有米其林推薦的美食？」`
    * **你的行動**: `search_mrt_food(station_name='東門', source_keyword='米其林')`

* **意圖: 查詢有哪些美食地圖**
    * **使用者**: `「你有哪幾種美食地圖？」`
    * **你的行動**: `list_available_food_maps()`

* **意圖: 查詢特定捷運線資訊**
    * **使用者**: `「可以告訴我板南線有哪些站嗎？」`
    * **你的行動**: `get_metro_line_info(line_name='板南線')`

* **意圖: 列出所有捷運線**
    * **使用者**: `「台北捷運總共有幾條線？」`
    * **你的行動**: `list_all_metro_lines()`

* **意圖: 列出所有車站**
    * **使用者**: `「把所有捷運站的名字都列給我」`
    * **你的行動**: `list_all_stations()`
---

**語言原則 (Language Protocol)**:
* 你的所有回覆，從第一個字到最後一個標點符號，都**必須**使用與使用者完全相同的語言。{language_instruction}
"""

# ---------------------------------------------------------------------
# 4. Agent 組裝 (Agent Assembly)
# ---------------------------------------------------------------------

# 建立 Agent 的 Prompt 模板，包含系統指令、歷史訊息、使用者輸入和 Agent 的思考過程。
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def get_language_instruction(lang_code: str) -> tuple[str, str]:
    """
    根據前端傳來的語言代碼，返回對應的語言名稱和給 LLM 的指令。
    這是實現多語言功能的關鍵。
    """
    instructions = {
        "en": ("English", "Your entire response must be in English."),
        "ja": ("日本語", "あなたの応答はすべて日本語でなければなりません。"),
        "zh-Hant": ("繁體中文", "你的所有回覆，包含任何問候或說明，都必須是繁體中文。")
    }
    # 如果找不到對應的語言，預設使用繁體中文
    return instructions.get(lang_code, instructions["zh-Hant"])

# 使用 LLM、工具集和 Prompt 模板來建立 Agent
agent = create_tool_calling_agent(llm, all_tools, prompt_template)

# 建立 Agent 執行器，這是實際運行 Agent 的物件
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True, # 設為 True，可以在後端終端機看到 Agent 的詳細思考過程，方便除錯
    handle_parsing_errors="抱歉，我好像有點理解錯誤，可以請您換個方式問我嗎？" # 當 AI 無法解析指令時的友善回覆
)