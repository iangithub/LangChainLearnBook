from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import dotenv_values

config = dotenv_values(".env")

model = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

# exp1
parser = StrOutputParser()
chain = model | parser
# model.invoke("你好").content
chain.invoke("你好")


# exp2
parser = StrOutputParser()
chain = model | parser
# model.invoke("你好").content
chain.invoke("將以下的內容翻譯為英文：知之為知之，不知為不知，是知也。")

# exp3
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "將以下的內容翻譯為{language}"), ("user", "{text}")]
)

chain = prompt_template | model | parser

target_language = "日文"
user_input = "知之為知之，不知為不知，是知也。"
chain.invoke({"language": target_language, "text": user_input})
