from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(".env")

aoai_api_key=os.getenv("AOAI_API_KEY")
aoai_endpoint=os.getenv("AOAI_ENDPOINT")
embed_deployment_name=os.getenv("AOAI_EMBED_DEPLOYMENT_NAME")

qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")

embeddings_model = AzureOpenAIEmbeddings(
    api_key=aoai_api_key,
    azure_deployment=embed_deployment_name, 
    openai_api_version="2024-06-01",
    azure_endpoint=aoai_endpoint,
)

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

collection_name = "my_collection"

qdrant = Qdrant(client, collection_name, embeddings_model)

result = qdrant.similarity_search_with_score(
    query = "本補助發放對象", 
    k = 1 , 
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.page", match=models.MatchValue(value=6)
            ),
        ]
    )
)

print(result)