from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import dotenv_values

config = dotenv_values(".env")

prompt = PromptTemplate.from_template("Translate the following English text to zh-tw: {text}")


# 初始化語言模型
model = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

# 建 LLMChain
chain = LLMChain(llm=model, prompt=prompt,verbose=True)

# 執行 LLMChain
#result = chain.invoke({"text": "Hello, how are you?"})
#result = chain.run(text="Hello, how are you?")
#result = chain.run({"text": "Hello, how are you?"})
result = chain(inputs={"text": "Hello, how are you?"})
print(result) 


