from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import dotenv_values

config = dotenv_values(".env")

# 初始化語言模型
llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

# 建立提示樣板
chinese_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一位短文寫作高手，將以使用者指定的主題進行寫作創作"), ("user", "{topic}")]
)

translation_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一位中英文語言專家，負責中文英的翻譯工作，翻譯的品質必須確保不可以失去文章內容原意，你的輸出結果必須符合以下格式\n\n 中文文章:..... ; 英文文章:...."), ("user", "{chinese_article}")]
)

# 使用 RunnableSequence 方式建立工作流程
work_flow = RunnableSequence(
    chinese_prompt,llm,translation_prompt,llm,StrOutputParser()
)

# 使用 LCEL 表達式建立工作流程
# work_flow = chinese_prompt | llm | translation_prompt | llm | StrOutputParser()

print(work_flow.invoke({"topic":"生成式AI的未來"}))