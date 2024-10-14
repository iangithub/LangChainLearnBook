# import modules
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import dotenv_values
config = dotenv_values(".env")


# get environment variables
deployment_name = config.get("AZURE_OPENAI_DEPLOYMENT_NAME")
api_key = config.get("AZURE_OPENAI_API_KEY")
azure_endpoint = config.get("AZURE_OPENAI_ENDPOINT")
api_version = config.get("AZURE_OPENAI_API_VERSION")

model = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    azure_deployment=deployment_name,
    openai_api_version=api_version,
)

parser = StrOutputParser()
chain = model | parser

messages = [
    SystemMessage(
        content="""
                  你是一個熱情的台灣人，使用繁體中文回答問題。
                  """
    ),
    HumanMessage(
        content="""
                  中壢有什麼好吃的？
                 """
    ),
]

result = chain.invoke(messages)
print(result)
