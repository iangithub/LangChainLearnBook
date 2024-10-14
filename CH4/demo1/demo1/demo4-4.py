# 引入Chain模組
from langchain.chains import SequentialChain,LLMChain

# 引入OpenAI LLM模組
from langchain_openai import AzureChatOpenAI

# 引入prompt模組
from langchain_core.prompts import PromptTemplate

import os
from dotenv import dotenv_values

config = dotenv_values(".env")

# 定義描述城市的提示樣板
describe_prompt = PromptTemplate(
    input_variables=["city"],
    template="請描述這個城市：### {city} ###"
)

# 定義描述氣候的提示樣板
climate_prompt = PromptTemplate(
    input_variables=["city"],
    template="請描述這個城市的夏天氣候：### {city} ###"
)

# 定義生成旅遊建議的提示樣板，根據城市描述和氣候描述
travel_prompt = PromptTemplate(
    input_variables=["description"],
    template="根據這個城市的描述和夏天氣候，為遊客提供一些旅遊指南：### {description} ### ，氣候：### {climate} ###"
)

# 定義翻譯成英文的提示樣板
translate_prompt = PromptTemplate(
    input_variables=["travel"],
    template="請將以下內容翻譯成英文：### {travel} ###"
)

llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"), 
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.9) 

# 描述城市的chain
describe_chain = LLMChain(llm=llm, prompt=describe_prompt, output_key="description")

# 建立描述氣候的 LLMChain
climate_chain = LLMChain(llm=llm, prompt=climate_prompt, output_key="climate")

# 生成旅遊建議的chain
travel_chain = LLMChain(llm=llm, prompt=travel_prompt, output_key="travel")

# 翻譯的chain
translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="final_advice")

# 建立 SequentialChain
sequential_chain = SequentialChain(
    chains=[describe_chain, climate_chain, travel_chain, translate_chain],
    input_variables=["city"],  # 初始輸入變數
    output_variables=["final_advice"]  # 最終輸出變數
)

result = sequential_chain.invoke("高雄")
print(result["final_advice"])