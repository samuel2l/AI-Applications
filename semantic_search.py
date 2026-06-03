# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")

# Create Pinecone index
pc.delete_index('pinecone-datacamp')
pc.create_index(
    name='pinecone-datacamp', 
    dimension=1536,
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

# Connect to index and print the index statistics
index = pc.Index("pinecone-datacamp")
print(index.describe_index_stats())



# Upserting vectors for semantic search
# Time to embed some text data and upsert the vectors and metadata into your 'pinecone-datacamp' index! You've be given a dataset named squad_dataset.csv, and a sample of 200 rows has been loaded in the DataFrame, df.

# In this exercise, to interact with the OpenAI API to use their embedding model, you don't need to create and use your own API key. A valid OpenAI client has been created for you and assigned to the client variable.

# Your task is to embed the text using OpenAI's API and upsert the embeddings and metadata into the Pinecone index under the namespace, squad_dataset.

# Instructions
# 100 XP
# Initialize the Pinecone client with your API key (the OpenAI client is already available as client).
# Extract the 'id', 'text', and 'title' metadata from each row in the batch.
# Encode texts using 'text-embedding-3-small' from OpenAI with dimensionality 1536.
# Upsert the vectors and metadatas to a namespace called 'squad_dataset'.
# Initialize the Pinecone client
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")
index = pc.Index('pinecone-datacamp')

batch_limit = 100

for batch in np.array_split(df, len(df) / batch_limit):
    # Extract the metadata from each row
    metadatas = [{
      "text_id": row['id'],
      "text": row['text'],
      "title": row['title']} for _, row in batch.iterrows()]
    
    texts = batch['text'].tolist()
    
    ids = [str(uuid4()) for _ in range(len(texts))]
    
    # Encode texts using OpenAI
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    embeds = [np.array(x.embedding) for x in response.data]
    
    # Upsert vectors to the correct namespace
    index.upsert(
        vectors=zip(ids, embeds, metadatas),
        namespace="squad_dataset"
    )



# Initialize the Pinecone client
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")
index = pc.Index('pinecone-datacamp')

query = "What is in front of the Notre Dame Main Building?"

# Create the query vector

query_response = client.embeddings.create(

    input=query,

    model="text-embedding-3-small"

)

query_emb = query_response.data[0].embedding

# Query the index and retrieve the top five most similar vectors

retrieved_docs = index.query(

    vector=query_emb,

    top_k=5,

    namespace="squad_dataset"

)

for result in retrieved_docs['matches']:

    print(f"{result['id']}: {round(result['score'], 2)}")

    print('\n')