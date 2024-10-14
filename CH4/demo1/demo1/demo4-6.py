# 引入必要的模組
from langchain.chains import LLMChain,ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI



import os
from dotenv import dotenv_values
config = dotenv_values(".env")

# 初始化 Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"), 
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.5) 


# 定義翻譯 chain
translate_template = """將以下中文文本翻譯成英文：
{input}"""

# 定義寫作 chain
write_template = """根據以下提示創作一段文字：
{input}"""

# 定義一般 chain
general_template = """回答以下問題：
{input}"""
general_prompt = PromptTemplate(
    template=general_template,
    input_variables=["input"],
    output_variables=["text"]
)
general_chain = LLMChain(llm=llm, prompt=general_prompt)

# 每一個任務Chain的資訊
prompt_infos = [
    {
        "name": "translate_chain", # chain 名稱
        "description": "進行中文翻譯成英文的任務", # chain 的簡單描述
        "prompt_template": translate_template, # chain的提示樣板
    },
    {
        "name": "write_chain", # chain 名稱
        "description": "進行創意寫作的任務", # chain 的簡單描述
        "prompt_template": write_template , # chain的提示樣板
    },
]

# 透過 prompt_infos 陣列取得各個 chain 的name以及description資訊
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# 使用 MULTI_PROMPT_ROUTER_TEMPLATE 格式化提示訊息
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

#  建立 LLMRouterChain 物件
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

print(router_prompt)

# 建立執行任務的chain物件
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# 建立 MultiPromptChain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=general_chain, # 預設執行chain
    verbose=True,
)

# 執行 MultiPromptChain
result =chain.invoke("請寫一篇關於那年夏天初戀的文章。");
# result =chain.invoke("翻譯這段話：10年前,我遇見了你,10年後,你遇見了我,於是我們一起遇見了彼此的未來。");
# result =chain.invoke("愛因斯坦是誰？");

print(result["text"])



