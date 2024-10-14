# 下面為 qdrant 進階使用技巧的參考程式碼，因為每個專案的需求不同，所以這裡只提供一個簡單的範例，讓讀者可以參考。
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv(".env")
qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

## 	
# 12-1 Qdrant多租戶的設計
##

# 1.	建立 collection，帶有 user_id 的 payload 
client.upsert(
    collection_name="my_collection",
    points=[
        models.PointStruct(
            id=1,
            payload={"user_id": "user_1"},
            vector=[0.9, 0.1, 0.1],
        ),
        models.PointStruct(
            id=2,
            payload={"user_id": "user_1"},
            vector=[0.1, 0.9, 0.1],
        ),
        models.PointStruct(
            id=3,
            payload={"user_id": "user_2"},
            vector=[0.1, 0.1, 0.9],
        ),
    ],
)

# 2.	進行搜尋，並且帶有 user_id 的條件
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.1, 0.9],
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value="user_1")
            )
        ]
    ),
    limit=10
)

# 3.	建立 collection 的索引

client.create_collection(
    collection_name="my_collection",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(
        payload_m=16,
        m=0,
    ),
)
client.create_payload_index(
    collection_name="my_collection",
    field_name="user_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

## 	
# 12-2 Qdrant 索引設計
##

# 1.    全文檢索的索引設計

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_payload_index(
    collection_name="my_collection",
    field_name="content",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=15,
        lowercase=True,
    ),
)

# 2.    參數化的索引設計

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_payload_index(
    collection_name="my_collection",
    field_name="age",
    field_schema=models.IntegerIndexParams(
        type=models.IntegerIndexType.INTEGER,
        lookup=False,
        range=True,
    ),
)

# 3.    集合的向量索引

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.update_collection(
    collection_name="my_collection",
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=100,
    )
)

# 4.    稀疏向量索引

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="my_collection",
    sparse_vectors={
        "text": models.SparseVectorIndexParams(
            index=models.SparseVectorIndexType(
                on_disk=False,
            ),
        ),
    },
)

## 	
# 12-3 Qdrant分散式部署
##


# 1.	建立 Qdrant 的分散式部署的配置檔案

# qdrant 部署的同個資料夾進，建立一個 config.yaml 檔。可以參考以下的配置的內容
'''
cluster:
  enabled: true
  p2p:
    port: 6335
  consensus:
    tick_period_ms: 100
'''

# 2.	Qdrant 的 sharding

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="my_collection",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    shard_number=6,
    replication_factor=2,
)

# 3.	Qdrant 的資料一致性 Write consistency factor

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(size=300, distance=models.Distance.COSINE),
    shard_number=6,
    replication_factor=2,
    write_consistency_factor=2,
)


# 4.	Qdrant 的資料一致性 Read Consistency Parameter

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.search(
    collection_name="{collection_name}",
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="city",
                match=models.MatchValue(
                    value="London",
                ),
            )
        ]
    ),
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    query_vector=[0.2, 0.1, 0.9, 0.7],
    limit=3,
    consistency="majority",
)

# 5.	Qdrant 的資料一致性 Write Order Paremeter

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.upsert(
    collection_name="{collection_name}",
    points=models.Batch(
        ids=[1, 2, 3],
        payloads=[
            {"color": "red"},
            {"color": "green"},
            {"color": "blue"},
        ],
        vectors=[
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
        ],
    ),
    ordering=models.WriteOrdering.STRONG,
)
