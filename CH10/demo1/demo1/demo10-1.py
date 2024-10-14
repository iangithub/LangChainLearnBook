from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

import os
from dotenv import dotenv_values
config = dotenv_values(".env")


# 初始化語言模型
generator_llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
)

embedding_llm = AzureOpenAIEmbeddings(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
)

# ----- 第一次要把知識文件加入Qdrant 向量資料庫時，執行以下程式碼 -----

# # Load PDF文件
# loader = PyPDFLoader("../docs/勞動基準法.pdf")
# pages = loader.load_and_split()

# # 分割文本
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(pages)

# # Qdrant向量資料庫
# qdrant = QdrantVectorStore.from_documents(
#     splits,
#     embedding=embedding_llm,
#     url="http://localhost:6333",  # 假設Qdrant運行在本地的6333端口
#     collection_name="km_docs",
# )

#---------------------------------------------------------




# ------- 後續查詢時，已有向量資料，請執行以下程式碼 -------

# Qdrant client
client = QdrantClient(url="http://localhost:6333")
collection_name = "km_docs"
qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_llm
    )

# -------------------------------------------------------


# 設置檢索器
retriever = qdrant.as_retriever(search_kwargs={"k": 3}) # 檢索前3個最相似的文檔

# 建立提示樣板
q_template = ChatPromptTemplate.from_template("""你是一位精通台灣勞基法的專家。請根據以下參考資料回答問題：

參考資料：{context}

問題：{question}

專家回答：""")

# 建立 QA Chain
qa_chain = (
    {
        "context": retriever ,
        "question": RunnablePassthrough(),
    }
    | q_template
    | generator_llm
    | StrOutputParser()
)


# 步驟 7: 進行查詢
response = qa_chain.invoke("勞工加班費的計算方式是什麼？")

print(response)
