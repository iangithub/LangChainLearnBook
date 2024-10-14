from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain,LLMChain
import os
from dotenv import dotenv_values

config = dotenv_values(".env")

# 定義描述城市的提示樣板
describe_prompt  = PromptTemplate(
    input_variables=["city"],
    template="請用一段優雅的文字描述這個城市：### {city} ###",
)

# 定義翻譯成英文的提示樣板
translate_prompt = PromptTemplate(
    input_variables=["description"],
    template="請將以下描述翻譯成英文：### {description} ###"
)

llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"), 
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.9) 

# 建立兩個 LLMChain，分別對應一個任務步驟
describe_chain = LLMChain(llm=llm, prompt=describe_prompt)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

chain = SimpleSequentialChain(chains=[describe_chain, translate_chain])

result = chain.invoke("高雄")
print(result["output"])

