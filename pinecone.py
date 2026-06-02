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

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

index = pc.Index('datacamp-index')
ids = ['2', '5', '8']

# Fetch the vectors from the connected Pinecone index
fetched_vectors = index.fetch(ids=ids)

# Extract the metadata from each result in fetched_vectors
metadatas = [fetched_vectors['vectors'][id]['metadata'] for id in ids]
print(metadatas)

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

index = pc.Index('datacamp-index')

# Retrieve the top three most similar records
query_result = index.query(

    vector=vector,

    top_k=3

)
print(query_result)


# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

# Create an index that uses the dot product distance metric
pc.delete_index("dotproduct-index")
pc.create_index(
    name="dotproduct-index",
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# Print a list of your indexes
print(pc.list_indexes())

# Initialize the Pinecone client using your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

index = pc.Index('datacamp-index')

# Retrieve the MOST similar vector with genre and year filters
query_result = index.query(
    vector=vector,
    top_k=1,
    filter={
        "genre":"thriller",
        "year":{"$lt":2018}
    }
)
print(query_result)

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")



index = pc.Index('datacamp-index')

# Update the values of vector ID 7

index.update(id="7", values=vector)

# Fetch vector ID 7

fetched_vector = index.fetch(ids=["7"])

print(fetched_vector)

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

index = pc.Index('datacamp-index')

# Update the metadata of vector ID 7
index.update(
    id="7",
    set_metadata={
        "genre": "thriller",
        "year": 2024
    }
)

# Fetch vector ID 7

fetched_vector = index.fetch(ids=["7"])

print(fetched_vector)

# Initialize the Pinecone client using your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

index = pc.Index('datacamp-index')

# Delete vectors
index.delete(ids=["3","4"])

# Retrieve metrics of the connected Pinecone index
print(index.describe_index_stats())