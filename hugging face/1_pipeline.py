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