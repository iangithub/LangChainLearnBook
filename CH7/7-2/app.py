# import modules
from openai import AzureOpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")

# Setting up the Azure OpenAI client
deployment_name = config.get("AZURE_OPENAI_DEPLOYMENT_NAME")
api_key = config.get("AZURE_OPENAI_API_KEY")
azure_endpoint = config.get("AZURE_OPENAI_ENDPOINT")
api_version = config.get("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version
)

# Call the OpenAI API

system_prompt = "你是一個多愁善感的詩人。使用繁體中文作答。"
user_question = "窗外下起雨了。"

try:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
    )
    # print the response
    print("Q: " + user_question)
    print("A: " + response.choices[0].message.content)

except Exception as e:
    # Handles all other exceptions
    print("An exception has occured.")
