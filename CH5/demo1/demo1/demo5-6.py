from langchain_core.runnables import RunnableSequence,RunnableWithFallbacks,RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import time

import os
from dotenv import dotenv_values

config = dotenv_values(".env")


# 定義提示樣版
advanced_prompt = ChatPromptTemplate.from_template("請回答以下問題：{question}")
base_prompt = ChatPromptTemplate.from_template("請回答以下問題：{question}")

# 定義模型
advanced_llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.5 
)
base_llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_Base_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.5 
)

# 進階模型Chain
advanced_chain = RunnableSequence(
    advanced_prompt,
    advanced_llm
)

# 基礎模型Chain
base_chain = RunnableSequence(
    base_prompt,
    base_llm
)

# 模擬不穩定隨機失敗的進階模型
def unstable_advanced_model(query):
    # 模擬隨機失敗
    if time.time() % 2 == 0:
        raise Exception("LLM Service unavailable")
    return advanced_chain.invoke(query)

# 預設失敗回應
def predefined_fallback(query):
    return "很抱歉，目前無法回應您的問題，請洽客服專線。"

# 建立問答Chain
qa_chain = RunnableLambda(unstable_advanced_model)

# 使用 RunnableWithFallbacks 建立問答系統
qa_system = RunnableWithFallbacks(
    runnable=qa_chain,
    fallbacks=[base_chain, RunnableLambda(predefined_fallback)]
)

# 測試問答系統
for _ in range(5):
    try:
        result = qa_system.invoke({"question": "什麼是生成式AI"})
        print(f"回答: {result.content}")
    except Exception as e:
        print(f"錯誤: {str(e)}")
    print()
    time.sleep(1)