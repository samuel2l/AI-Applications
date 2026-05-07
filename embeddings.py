# Embeddings are numerical vector representations of data (usually text) that capture semantic meaning.

# Instead of treating words or sentences as plain text, embeddings convert them into lists of numbers so machines can understand relationships and similarity.

# For example:

# * “I love dogs”
# * “I like puppies”

# will produce embeddings that are close together in vector space because they mean similar things.
# Common uses:

# * Semantic search
# * Chatbots and RAG systems
# * Recommendations
# * Clustering similar documents
# * Duplicate detection
# * Classification

# Example:
# If a user searches:

# “How do I build a Flutter app?”

# you can embed the query and compare it against embeddings of stored documents to find the most semantically relevant content, even if the wording differs.

# Modern embedding models like text-embedding-3-small generate high-dimensional vectors optimized for semantic understanding.
# Create an OpenAI client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="This is me testing how embeddings work"
)

# Convert the response into a dictionary
response_dict = response.model_dump()
print(response_dict)