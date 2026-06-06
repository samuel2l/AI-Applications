# Initialize the Pinecone client
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")
index = pc.Index('pinecone-datacamp')

batch_limit = 100

for batch in np.array_split(youtube_df, len(youtube_df) / batch_limit):
    # Extract the metadata from each row
    metadatas = [{
      "text_id": row['id'],
      "text": row['text'],
      "title": row['title'],
      "url": row['url'],
      "published": row['published']} for _, row in batch.iterrows()]
    
    texts = batch['text'].tolist()
    
    ids = [str(uuid4()) for x in range(len(texts))]
    
    # Encode texts using OpenAI
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeds = [np.array(x.embedding) for x in response.data]
    
    # Upsert vectors to the correct namespace
    index.upsert(
        vectors=zip(ids, embeds, metadatas),
        namespace='youtube_rag_dataset'
    )
    
print(index.describe_index_stats())


# Initialize the Pinecone client
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")
index = pc.Index('pinecone-datacamp')

# Building a retrieval function
# A key process in the Retrieval Augmented Generation (RAG) workflow is retrieving data from the database. In this exercise, you'll design a custom function called retrieve() that will perform this crucial process in the final exercise of the course.

# Instructions
# 100 XP
# Initialize the Pinecone client with your API key (the OpenAI client is available as client).
# Define the function retrieve that takes four parameters: query, top_k, namespace, and emb_model.
# Embed the input query using the emb_model argument.
# Retrieve the top_k similar vectors to query_emb with metadata, specifying the namespace provided to the function as an argument
# Define a retrieve function that takes four arguments: query, top_k, namespace, and emb_model
def retrieve(query, top_k, namespace, emb_model):
    # Encode the input query using OpenAI
    query_response = client.embeddings.create(
        input=query,
        model=emb_model
    )
    
    query_emb = query_response.data[0].embedding
    
    # Query the index using the query_emb
    docs = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    
    retrieved_docs = []
    sources = []
    for doc in docs['matches']:
        retrieved_docs.append(doc['metadata']['text'])
        sources.append((doc['metadata']['title'], doc['metadata']['url']))
    
    return retrieved_docs, sources

documents, sources = retrieve(
  query="How to build next-level Q&A with OpenAI",
  top_k=3,
  namespace='youtube_rag_dataset',
  emb_model="text-embedding-3-small"
)

print(documents)
print(sources)



# RAG questions answering function
# You're almost there! The final piece in the RAG workflow is to integrate the retrieved documents with a question-answering model.

# A prompt_with_context_builder() function has already been defined and made available to you. This function takes the documents retrieved from the Pinecone index, and integrates them into a prompt that the question-answering model can ingest:

# def prompt_with_context_builder(query, docs):
#     delim = '\n\n---\n\n'
#     prompt_start = 'Answer the question based on the context below.\n\nContext:\n'
#     prompt_end = f'\n\nQuestion: {query}\nAnswer:'

#     prompt = prompt_start + delim.join(docs) + prompt_end
#     return prompt
# You'll implement the question_answering() function, which will provide OpenAI's language model gpt-4o-mini with additional context and sources with which it can answer your questions.

# Instructions
# 100 XP
# Initialize the Pinecone client with your API key (the OpenAI client is available as client).
# Retrieve the three most similar documents to the query text from the 'youtube_rag_dataset' namespace.
# Generate a response to the provided prompt and sys_prompt using OpenAI's 'gpt-4o-mini' model, specified using the chat_model function argument.

# Initialize the Pinecone client
pc = Pinecone(api_key="pcsk_5dVpiR_Gjgx2xBfPbBPvKZGsd8UXSuTw6fuFn6WaDuCAfpMfiSDQYCUPFwyiVeqgYcfAVr")
index = pc.Index('pinecone-datacamp')

query = "How to build next-level Q&A with OpenAI"

# Retrieve the top three most similar documents and their sources
documents, sources = retrieve(
    query,
    top_k=3,
    namespace='youtube_rag_dataset',
    emb_model="text-embedding-3-small"
)

prompt_with_context = prompt_with_context_builder(query, documents)
print(prompt_with_context)

def question_answering(prompt, sources, chat_model):
    sys_prompt = "You are a helpful assistant that always answers questions."
    
    # Use OpenAI chat completions to generate a response
    res = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    answer = res.choices[0].message.content.strip()
    answer += "\n\nSources:"
    for source in sources:
        answer += "\n" + source[0] + ": " + source[1]
    
    return answer

answer = question_answering(
  prompt=prompt_with_context,
  sources=sources,
  chat_model='gpt-4o-mini'
)

print(answer)