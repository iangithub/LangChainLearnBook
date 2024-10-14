from langchain_core.runnables import RunnableBranch,RunnableSequence
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
    temperature=0.3 # 使用較低的溫度值以獲得更精確的答案
)

# 語言識別Chain
language_identification_prompt= ChatPromptTemplate.from_template(
"Please identify the language of the following text. "
"Respond with 'Chinese' for Chinese, 'English' for English, or 'Other' for any other language. "
"Text: {text}")
language_identification_chain = language_identification_prompt | llm | StrOutputParser()

# 建立中文服務Chain
chinese_prompt = ChatPromptTemplate.from_template("你是一位中文客服機器人，請根據用戶的問題提供中文回應。###{text}")
chinese_chain = chinese_prompt | llm | StrOutputParser()

# 建立英文服務Chain
english_prompt = ChatPromptTemplate.from_template("You are an English customer service bot. Please respond to the user's query in English.###{text}")
english_chain = english_prompt | llm | StrOutputParser()



# 建立 RunnableBranch
workflow = RunnableSequence(
    {"language": language_identification_chain,
      "text": lambda x: x["text"]},
    RunnableBranch(
        (lambda x: x["language"].strip().lower() == "chinese"
         , chinese_chain),
        (lambda x: x["language"].strip().lower() == "english"
         , english_chain),
        english_chain
    )
)

# text = "上個星期入住了這家飯店，整體感覺還不錯，服務人員態度很好，房間也很乾淨，下次還會再來。"
# text = "I stayed in this hotel last week. The overall feeling is pretty good. The service staff is very friendly and the room is very clean. I will come back next time."
text = "先週このホテルに泊まりました。全体的な雰囲気はとても良く、スタッフはとてもフレンドリーで、次回もまた来ます。"
results = workflow.invoke({"text": text})

print(results)