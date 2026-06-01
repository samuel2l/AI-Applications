# Import ServerlessSpec
from pinecone import ServerlessSpec
from pinecone import Pinecone
# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

# Create your Pinecone index
pc.delete_index('my-first-index')
pc.create_index(
    name="my-first-index",
    dimension=256,
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)


 # Set up the client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

# Connect to your index
index = pc.Index("my-first-index")

# Print the index statistics
print(index.describe_index_stats())

# vectors = [
#         {
#         "id": "0",
#         "values": [0.025525547564029694, ..., 0.0188823901116848]
#         "metadata": {"genre": "action", "year": 2024}
#     }
# ]

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

# Connect to your index
index = pc.Index("datacamp-index")

# Ingest the vectors and metadata
index.upsert(vectors)

# Print the index statistics
print(index.describe_index_stats())