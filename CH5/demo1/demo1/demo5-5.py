from langchain_core.runnables import RunnableSequence,RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json

import os
from dotenv import dotenv_values

config = dotenv_values(".env")

# 初始化語言模型
llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.3 
)


def validate_order(order):
    """
    驗證訂單資訊
    """
    errors = []
    if not order.get("customer_id"):
        errors.append("缺少客戶ID")
    if not order.get("items") or len(order["items"]) == 0:
        errors.append("訂單中沒有商品")
    return {"order": order, "is_valid": len(errors) == 0, "errors": errors}

validate_order_RunnableLambda = RunnableLambda(validate_order)

def prepare_llm_input(processed_order):
    """
    準備 LLM 輸入
    """
    return {"order_info": json.dumps(processed_order, ensure_ascii=False)}

prepare_llm_input_RunnableLambda = RunnableLambda(prepare_llm_input)


# 建立 LLM 摘要Chain
summary_prompt= ChatPromptTemplate.from_template(
        "你是一個電子商務平台的客戶服務助手。請根據以下訂單內容生成訂單摘要。"
        "如果訂單無效，請解釋原因。訂單內容：### {order_info} ### "
    )
summary_chain =summary_prompt | llm

# 建立訂單處理工作流程
workflow = RunnableSequence(
    validate_order,
    prepare_llm_input_RunnableLambda,
    summary_chain,
    StrOutputParser()
)

# 測試工作流程
test_orders = [
    {
        "customer_id": "CUS001",
        "items": [
            {"name": "筆記本電腦", "price": 35000, "quantity": 1},
            {"name": "滑鼠", "price": 2500, "quantity": 2}
        ]
    },
    {
        "customer_id": "CUS002",
        "items": []
    },
    {
        "items": [
            {"name": "鍵盤", "price": 500, "quantity": 1}
        ]
    }
]

for order in test_orders:
    result = workflow.invoke(order)
    print(result)
    print("--------------------------------------------------")
