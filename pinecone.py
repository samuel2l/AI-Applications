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