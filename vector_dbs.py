# Create a persistant client
client = chromadb.PersistentClient()

# Create a netflix_title collection using the OpenAI Embedding function
collection = client.create_collection(
    name="netflix_titles",
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key="<OPENAI_API_TOKEN>"
    )
)

# List the collections
print(client.list_collections())


# Load the encoder for the OpenAI text-embedding-3-small model
enc = tiktoken.encoding_for_model("text-embedding-3-small")

# Encode each text in documents and calculate the total tokens
total_tokens = sum(len(enc.encode(doc)) for doc in documents)

cost_per_1k_tokens = 0.00002

# Display number of tokens and cost
print('Total tokens:', total_tokens)
print('Cost:', (total_tokens / 1000) * cost_per_1k_tokens)


# Recreate the netflix_titles collection
collection = client.create_collection(
    name="netflix_titles",
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key="<OPENAI_API_TOKEN>"
    )
)

# Add the documents and IDs to the collection
collection.add(
    documents=documents,
    ids=ids
)

# Print the collection size and first ten items
print(f"No. of documents: {collection.count()}")
print(f"First ten documents: {collection.peek(10)}")

# Retrieve the netflix_titles collection
collection = client.get_collection(
    name="netflix_titles",
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key="<OPENAI_API_TOKEN>"
    )
)

# Query the collection for "films about dogs"
result = collection.query(
    query_texts=["films about dogs"],
    n_results=3
)

print(result)