# 引入Chain模組
from langchain.chains import LLMChain

# 引入OpenAI LLM模組
from langchain_openai import AzureChatOpenAI

# 引入prompt模組
from langchain_core.prompts import PromptTemplate

import os
from dotenv import dotenv_values

config = dotenv_values(".env")

llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"), 
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.5) 

# 定義情緒分析的提示樣板
sentiment_analysis_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="根據這段話分析情緒，並僅回答 'positive' 或 'negative'：'{user_input}'"
)
# 建立情緒分析的 LLMChain
sentiment_analysis_chain = LLMChain(llm=llm, prompt=sentiment_analysis_prompt)

# 負面情緒應對的 PromptTemplate
negative_response_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="使用者說了這段話：'{user_input}'。請給出一段安撫的回應。"
)
negative_response_chain = LLMChain(llm=llm, prompt=negative_response_prompt)

# 正面情緒應對的 PromptTemplate
positive_response_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="使用者說了這段話：'{user_input}'。請給出一段正向互動的回應。"
)
positive_response_chain = LLMChain(llm=llm, prompt=positive_response_prompt)


def execute_conditional_chain(user_input):
    # 第一步：使用 LLM 來分析情緒
    sentiment_result = sentiment_analysis_chain.run({"user_input": user_input})
    
    # 第二步：根據情緒結果選擇要執行的chain
    if sentiment_result.strip().lower() == "negative":
        # 如果情緒為負面，執行負面應對chain
        return negative_response_chain.invoke({"user_input": user_input})
    else:
        # 如果情緒為正面，執行正面應對鏈結
        return positive_response_chain.invoke({"user_input": user_input})

# 執行 Conditional Chain
result = execute_conditional_chain("我對於你們的服務感到非常滿意，服務人員很用心，環境也很整潔。")

print(result["text"])

