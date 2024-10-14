from langchain_openai import AzureChatOpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")

model = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate

example_prompt = HumanMessagePromptTemplate.from_template(
    "{description}"
) + AIMessagePromptTemplate.from_template("{classification}")

examples = [
    {
        "description": "食物偏甜",
        "classification": "南部人",
    },
    {
        "description": "食物偏鹹",
        "classification": "北部人",
    },
    {
        "description": "滷肉飯",
        "classification": "北部人",
    },
    {
        "description": "肉燥飯",
        "classification": "南部人",
    },
    {
        "description": "搭乘大眾運輸，不怕走路",
        "classification": "北部人",
    },
    {
        "description": "騎摩托車，不待轉",
        "classification": "南部人",
    },
    {
        "description": "講話婉轉，不直接",
        "classification": "北部人",
    },
    {
        "description": "講話直接",
        "classification": "南部人",
    },
]


from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

from langchain_core.prompts.chat import ChatPromptTemplate

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "請根據以下描述，判斷是哪一種人："),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)


chain = final_prompt | model | parser

user_input = "醬油喜歡有甜味"
# user_input = "熱情大方，講話直接"
response = chain.invoke({"input": user_input})
print("描述：", user_input)
print("分類：", response)
