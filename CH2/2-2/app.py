from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import dotenv_values

config = dotenv_values(".env")

model = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

parser = StrOutputParser()
chain = model | parser

user_input = "知之為知之，不知為不知，是知也。"

messages = [
    SystemMessage(content="將以下的內容翻譯為英文。"),
    HumanMessage(content=user_input),
]

chain.invoke(messages)
