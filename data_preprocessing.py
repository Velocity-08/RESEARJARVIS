from transformers import GPT2Tokenizer, BertTokenizer
import json

# Define file paths
input_file = "filtered_dataset.json"  # Input: Filtered dataset
output_file = "tokenized_dataset.json"  # Output: Tokenized dataset

# Initialize tokenizers
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        filtered_data = json.load(infile)

    tokenized_data = []

    for entry in filtered_data:
        query = entry["query"]
        response = entry["response"]

        # Tokenize using GPT tokenizer
        gpt_query_tokens = gpt_tokenizer.encode(query, add_special_tokens=True)
        gpt_response_tokens = gpt_tokenizer.encode(response, add_special_tokens=True)

        # Tokenize using BERT tokenizer
        bert_query_tokens = bert_tokenizer.encode(query, add_special_tokens=True)
        bert_response_tokens = bert_tokenizer.encode(response, add_special_tokens=True)

        # Append tokenized results
        tokenized_data.append({
            "query": query,
            "response": response,
            "gpt_query_tokens": gpt_query_tokens,
            "gpt_response_tokens": gpt_response_tokens,
            "bert_query_tokens": bert_query_tokens,
            "bert_response_tokens": bert_response_tokens
        })

    # Save tokenized data to output JSON file
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(tokenized_data, outfile, indent=4)

    print(f"Tokenized data saved to {output_file}")

# Run the tokenization process
tokenize_dataset(input_file, output_file)
