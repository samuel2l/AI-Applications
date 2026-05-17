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

#Embeddings always return 1536 numbers no matter the input size, which allows for consistent vector representations regardless of the length of the text.

#  Create an OpenAI client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="This is me testing how embeddings work"
)

# Convert the response into a dictionary
response_dict = response.model_dump()
print(response_dict)


#now we need to reduce the 1536-dimensional embedding vector to a lower dimension for visualization or further processing. We can use techniques like PCA (Principal Component Analysis) or t-SNE (t-distributed Stochastic Neighbor Embedding) for this purpose.

# Reduce the number of embeddings dimensions from 1,536 to two using the tsne model provided.
# Create reviews and embeddings lists using list comprehensions
categories = [product['category'] for product in products]
embeddings = [product['embedding'] for product in products]

# Reduce the number of embeddings dimensions to two using t-SNE
tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))


def create_embeddings(texts):
  response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]


# Finding the most similar product
# Being able to compute similarity between embeddings is a key step within embeddings applications. In this exercise, you'll return to the products list of dictionaries that you worked with previously, which contains the embedded short descriptions you also created earlier.

# You'll compare a piece of text to these embedded descriptions to identify the most similar description.
# Embed the search text
search_text = "soap"
search_embedding = create_embeddings(search_text)[0]

distances = []
for product in products:
  # Compute the cosine distance for each product description
  dist = distance.cosine(search_embedding, product['embedding'])
  distances.append(dist)

# Find and print the most similar product short_description    
min_dist_ind = np.argmin(distances)
print(products[min_dist_ind]['short_description'])

# omo