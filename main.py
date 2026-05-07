from transformers import pipeline 

gpt2_pipeline = pipeline(task="text-generation", model="openai-community/gpt2")

# Generate three text outputs with a maximum length of 10 tokens
results = gpt2_pipeline("What if AI", max_new_tokens=10, num_return_sequences=3)

for result in results:
    print(result['generated_text'])


#example output:#What if AI could help us solve some of the world's biggest problems? 
#What if AI could help us solve some of the world's biggest problems? 


#loading a dataset

# Load the "validation" split of the TIGER-Lab/MMLU-Pro dataset
from datasets import load_dataset
my_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

# Display dataset details
print(my_dataset)

#wikipedia is var you have loaded dataset into
# Filter the documents

# Filter the dataset for rows with the term "football" in the text column and save as filtered.
filtered = wikipedia.filter(lambda row: "football" in row["text"])

# Create a sample dataset
example = filtered.select(range(1))

print(example[0]["text"])


# available tasks are ['audio-classification', 'automatic-speech-recognition', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-text-to-text', 'image-to-image', 'image-to-text', 'mask-generation', 'ner', 'object-detection', 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 'text2text-generation', 'token-classification', 'translation', 'video-classification', 'visual-question-answering', 'vqa', 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection', 'translation_XX_to_YY']
#grammar checker
# Create a pipeline for grammar checking
grammar_checker = pipeline(
  task="text-classification", 
  model="abdulmatinomotoso/English_Grammar_Checker"
)

# Check grammar of the input text
output = grammar_checker("I will walk dog")
print(output)



#QNLI - Question Natural Language Inference. It is a popular dataset and task used in machine learning to evaluate the performance of natural language processing (NLP) models, specifically in understanding the relationship between a question and a context sentence.
# Create the pipeline
classifier = pipeline(task="text-classification", model="cross-encoder/qnli-electra-base")

# Predict the output
output = classifier("Where is the capital of France?, Brittany is known for its stunning coastline.")

print(output)


#DYNAMIC category assignment
text = "AI-powered robots assist in complex brain surgeries with precision."

# Create the pipeline
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

# Create the categories list
categories = ["politics", "science", "sports"]

# Predict the output
output = classifier(text, categories)

# Print the top label and its score
print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")


#Summarization - here are two main types: extractive, which selects key sentences from the original text, and abstractive, which generates new sentences summarizing main ideas.

#extractive summarization
# Create the summarization pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

# Summarize the text
summary_text = summarizer(original_text)

# Compare the length
print(f"Original text length: {len(original_text)}")
print(f"Summary length: {len(summary_text[0]['summary_text'])}")


#adjusting summary length
# Generate a summary of original_text between 1 and 10 tokens
short_summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", min_new_tokens=1, max_new_tokens=10)

short_summary_text = short_summarizer(original_text)

print(short_summary_text[0]["summary_text"])


# auto classes
#when to use normal pipelines and when to use auto classes
#PIPELINES
# I want to quickly compare multiple models for text generation tasks
# I need a quick way to classify customer reviews as positive or negative
# The task is a simple text summarization for news articles

# #AUTOCLASSES
# The task requires tokenizing financial reports, adding custom tokens for terms like 'EBITDA' or 'ROl'
# Our customer support model should prioritize the 'Urgent' category more often
# Import necessary library for tokenization
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Split input text into tokens
tokens = tokenizer.tokenize("AI: Making robots smarter and humans lazier!")

# Display the tokenized output
print(f"Tokenized output: {tokens}")


#full pipeline using auto classes eg
# Download the model and tokenizer
my_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
my_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Create the pipeline
my_pipeline = pipeline(task="sentiment-analysis", model=my_model, tokenizer=my_tokenizer)

# Predict the sentiment
output = my_pipeline("This course is pretty good, I guess.")
print(f"Sentiment using AutoClasses: {output[0]['label']}")

#question answering using pypdf
from pypdf import PdfReader

# Extract text from the PDF
reader = PdfReader("US_Employee_Policy.pdf")

# Extract text from all pages
document_text = ""
for page in reader.pages: 
    document_text += page.extract_text()

print(document_text)
# Load the question-answering pipeline
qa_pipeline = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad")

question = "What is the notice period for resignation?"

# Get the answer from the QA pipeline
result = qa_pipeline(question=question, context=document_text)

# Print the answer
print(f"Answer: {result['answer']}")



#retrying failed API calls with tenacity

# Import the tenacity library
from tenacity import retry,wait_random_exponential,stop_after_attempt

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Add the appropriate parameters to the decorator

@retry(wait=wait_random_exponential(min=5, max=40), stop=stop_after_attempt(4))
def get_response(model, message):
    response = client.chat.completions.create(
      model=model,
      messages=[message]
    )
    return response.choices[0].message.content
print(get_response("gpt-4o-mini", {"role": "user", "content": "List ten holiday destinations."}))


# Batching messages
# You are developing a fitness application to track running and cycling training, but find out that all your customers' distances have been measured in kilometers, and you'd like to have them also converted to miles.

# You decide to use the OpenAI API to send requests for each measurement, but want to avoid using a for loop that would send too many requests. You decide to send the requests in batches, specifying a system message that asks to convert each of the measurements from kilometers to miles and present the results in a table containing both the original and converted measurements.

# The measurements list (containing a list of floats) and the get_response() function have already been imported.

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

messages = []
# Provide a system message and user messages to send the batch
messages.append({
    "role":"system",
    "content": "Convert the following measurements from kilometers into miles. Present the results in a table containing both the old and new values"
})
# Append measurements to the message
[messages.append({
    "role":"user","content":f"{i}"
}) for i in measurements]

response = get_response(messages)
print(response)

# Setting token limits
# An e-commerce platform just hired you to improve the performance of their customer service bot built using the OpenAI API. You've decided to start by ensuring that the input messages do not cause any rate limit issue by setting a limit of 100 tokens, and test it with a sample input message.

# The tiktoken library has been preloaded.

client = OpenAI(api_key="<OPENAI_API_TOKEN>")
input_message = {"role": "user", "content": "I'd like to buy a shirt and a jacket. Can you suggest two color pairings for these items?"}
# Use tiktoken to create the encoding for your model
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = encoding.encode(input_message["content"])
# Check for the number of tokens
num_tokens = len(tokens)
print(num_tokens)
# Run the chat completions function and print the response
if num_tokens <= 100:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[input_message])
    print(response.choices[0].message.content)
else:
    print("Message exceeds token limit")


#tool calling

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

response= client.chat.completions.create(
    model="gpt-4o-mini",
    # Add the message
    messages=message_listing,
    # Add your function definition
    tools=function_definition
)

# Print the response
print(response.choices[0].message.tool_calls[0].function.arguments)


#specifying response format
# Create the OpenAI client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create the request
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
   {"role": "user", "content": "I have these notes with book titles and authors: New releases this week! The Beholders by Hester Musson, The Mystery Guest by Nita Prose. Please organize the titles and authors in a json file."}
  ],
  response_format = {"type":"json_object"}
)

# Print the response
print(response)

#moderation -  used to detect and filter out harmful or inappropriate content in user-generated text, ensuring a safer and more positive online environment. It can be used to identify and block content that violates community guidelines, such as hate speech, harassment, or explicit material. By implementing moderation tools, platforms can maintain a respectful and inclusive space for users while preventing the spread of harmful content.
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

message = "Can you show some example sentences in the past tense in French?"

# Use the moderation API
moderation_response = client.moderations.create(
    input=message
)

# print(moderation_response)
for flag in moderation_response.results[0].categories:
    print(flag)

