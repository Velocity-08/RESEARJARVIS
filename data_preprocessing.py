import json
from transformers import BertTokenizer  # Replace with your preferred tokenizer (e.g., GPT, custom)

# Define file paths
input_file = "filtered_dataset.json"  # Input: Filtered dataset
output_file = "preprocessed_dataset.json"  # Output: Preprocessed dataset

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Pretrained tokenizer

# Function to preprocess dataset
def preprocess_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        filtered_data = json.load(infile)

    preprocessed_data = []

    for entry in filtered_data:
        # Tokenize 'query' and 'response'
        query_tokens = tokenizer.tokenize(entry["query"])
        response_tokens = tokenizer.tokenize(entry["response"])
        
        # Convert tokens to IDs (numerical format)
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        response_ids = tokenizer.convert_tokens_to_ids(response_tokens)

        # Store preprocessed data
        preprocessed_data.append({
            "query_tokens": query_tokens,
            "response_tokens": response_tokens,
            "query_ids": query_ids,
            "response_ids": response_ids
        })

    # Save preprocessed data to output JSON file
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(preprocessed_data, outfile, indent=4)

    print(f"Preprocessed data saved to {output_path}")

# Run the preprocessing process
preprocess_dataset(input_file, output_file)
