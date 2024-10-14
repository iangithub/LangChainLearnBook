from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

api_token = os.getenv("HF_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=50,
    do_sample=False,
    huggingfacehub_api_token = api_token
)

response = llm.invoke("世界上最高的山是哪一座？")

print(response)