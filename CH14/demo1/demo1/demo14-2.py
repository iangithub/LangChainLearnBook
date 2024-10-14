
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from typing import List, Dict
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import dotenv_values

config = dotenv_values(".env")


# 初始化語言模型
llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.2
)

# 模擬房間的可用資料
rooms_availability: List[Dict] =  [
    {"roomno":"001","roomtype":"雙人房","available_date":"2024/9/1"},
    {"roomno":"001","roomtype":"雙人房","available_date":"2024/9/2"},
    {"roomno":"002","roomtype":"單人房","available_date":"2024/9/1"},
    {"roomno":"002","roomtype":"單人房","available_date":"2024/9/3"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/9/1"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/9/2"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/9/3"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/8/26"},
    {"roomno":"003","roomtype":"雙人房","available_date":"2024/8/27"}
]

# 取得當前日期
@tool
def get_current_date() -> str:
    """
    取得今天日期。

    返回:
    str: 今天日期，格式為 YYYY/MM/DD
    """
    return datetime.now().strftime("%Y/%m/%d")

# 查詢指定日期的可用房間
@tool
def check_room_availability(date: str) -> str:
    """
    查詢指定日期的可用房間。

    參數:
    date (str): 查詢日期，格式為 YYYY/MM/DD

    返回:
    str: 可用房間的資訊，如果沒有可用房間則返回無可預訂空房的訊息
    """
    try:
        # 驗證日期格式
        query_date = datetime.strptime(date, "%Y/%m/%d")
    except ValueError:
        return "日期格式不正確，請使用 YYYY/MM/DD 格式。"

    available_rooms = [
            room for room in rooms_availability 
            if datetime.strptime(room["available_date"], "%Y/%m/%d").date() == query_date.date()
        ]

    if not available_rooms:
        return f"抱歉，{date} 沒有可預訂的房間。"

    result = f"{date} 可預訂的房間如下：\n"
    for room in available_rooms:
        result += f"房間號碼：{room['roomno']}，類型：{room['roomtype']}\n"

    return result

# 輸出LLM回應過程
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# 設定工具
tools = [get_current_date,check_room_availability]

# 加入 chat memory
memory = MemorySaver()

# 建立Agent
agent = create_react_agent(llm, tools=tools, checkpointer=memory)

# Agent 啟動
config = {"configurable": {"thread_id": "168"}}
inputs = {"messages": [("user", "可以預訂明天住宿嗎")]}
print_stream(agent.stream(inputs, config=config, stream_mode="values"))

inputs = {"messages": [("user", "我剛預訂了什麼時候的房間")]}
print_stream(agent.stream(inputs, config=config, stream_mode="values"))
