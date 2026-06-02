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


#BATCHING

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    # Convert the iterable into an iterator
    it = iter(iterable)
    # Slice the iterator into chunks of size batch_size
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        # Yield the chunk
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))



# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

index = pc.Index('datacamp-index')

# Upsert vectors in batches of 100
for chunk in chunks(vectors):
    index.upsert(vectors=chunk)

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())


# Initialize the Pinecone client to allow 20 simultaneous requests.
# Upsert the vectors in vectors in batches of 200 vectors per request asynchronously, configuring 20 simultaneous requests.
# Print the updated metrics of the 'datacamp-index' Pinecone index.

# Initialize the client
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr", pool_threads=20)

index = pc.Index('datacamp-index',)

# Upsert vectors in batches of 200 vectors
with pc.Index('datacamp-index', pool_threads=20) as index:
    async_results = [
        index.upsert(vectors=chunk, async_req=True)
        for chunk in chunks(vectors, batch_size=200)
    ]
    [async_result.get() for async_result in async_results]

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())



# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")
index = pc.Index('datacamp-index')

# Upsert vector_set1 to namespace1
index.upsert(
    vectors=vector_set1,
    namespace="namespace1"
)

# Upsert vector_set2 to namespace2
index.upsert(
    vectors=vector_set2,
    namespace="namespace2"
)

# Print the index statistics
print(index.describe_index_stats())