from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv("./.env")

aoai_api_key=os.getenv("AOAI_API_KEY")
aoai_endpoint=os.getenv("AOAI_ENDPOINT")
embed_deployment_name=os.getenv("AOAI_EMBED_DEPLOYMENT_NAME")
gpt_deployment_name=os.getenv("AOAI_GPT_DEPLOYMENT_NAME")

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

collection_name = "subsidy_qa"
qdrant = Qdrant(client, collection_name, embeddings_model)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

model = AzureChatOpenAI(
    api_key=aoai_api_key,
    openai_api_version="2024-06-01",
    azure_deployment=gpt_deployment_name,
    azure_endpoint=aoai_endpoint,
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Add typing for input
class Question(BaseModel):
    input: str


rag_chain = retrieval_chain.with_types(input_type=Question)
