#以筆者們的上一本書「極速ChatGPT開發者兵器指南」做為retrival資料來源的範例
#https://www.drmaster.com.tw/bookinfo.asp?BookID=MP22359


from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv("./.env")

aoai_api_key=os.getenv("AOAI_API_KEY")
aoai_endpoint=os.getenv("AOAI_ENDPOINT")
embed_deployment_name=os.getenv("AOAI_EMBED_DEPLOYMENT_NAME")
gpt_deployment_name=os.getenv("AOAI_GPT_DEPLOYMENT_NAME")

qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")

loader = WebBaseLoader("https://www.drmaster.com.tw/bookinfo.asp?BookID=MP22359")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

model = AzureChatOpenAI(
    api_key=aoai_api_key,
    openai_api_version="2024-06-01",
    azure_deployment=gpt_deployment_name,
    azure_endpoint=aoai_endpoint,
    temperature=0,
)

embeddings_model = AzureOpenAIEmbeddings(
    api_key=aoai_api_key,
    azure_deployment=embed_deployment_name, 
    openai_api_version="2024-06-01",
    azure_endpoint=aoai_endpoint,
)

qdrant = Qdrant.from_documents(
    docs,
    embeddings_model,
    url=qdrant_url, 
    api_key=qdrant_api_key,
    collection_name="book",
    force_recreate=True,
)

retriever = qdrant.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "請回答依照 context 裡的資訊來回答問題:{context}。問題{input}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
    ])

document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///./langchain.db")

chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    output_messages_key="answer",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "1"}}


response = chain_with_history.invoke({"input": "請問這本書的作者？"}, config=config)
print(response["answer"])

response = chain_with_history.invoke({"input": "我剛剛的問題是什麼"}, config=config)
print(response["answer"])
