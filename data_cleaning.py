import json
import re

# Define file paths
input_file = "raw_dataset.json"  # Input: Raw dataset in JSON format
output_file = "cleaned_dataset.json"  # Output: Cleaned dataset in JSON format

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters and numbers (optional, adjust as needed)
    text = re.sub(r"[^a-zA-Z\s.,!?']", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Convert to lowercase
    text = text.lower()
    return text

# Process the dataset
def clean_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    cleaned_data = []

    for entry in raw_data:
        # Clean the 'query' and 'response' fields
        if "query" in entry:
            entry["query"] = clean_text(entry["query"])
        if "response" in entry:
            entry["response"] = clean_text(entry["response"])
        
        cleaned_data.append(entry)

    # Save cleaned data to output JSON file
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(cleaned_data, outfile, indent=4)

    print(f"Cleaned data saved to {output_path}")

# Run the cleaning process
clean_dataset(input_file, output_file)
