from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

api_token = os.getenv("HF_API_TOKEN")

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model= "mixedbread-ai/mxbai-embed-large-v1",
    task="feature-extraction",
    huggingfacehub_api_token=api_token,
)

texts = ["Hello, world!", "世界上最高的山是哪一座？"]

response = hf_embeddings.embed_documents(texts)

print(response)