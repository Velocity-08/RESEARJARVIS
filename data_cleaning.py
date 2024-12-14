import re
import json

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
    try:
        # Read the raw dataset from JSON file
        with open(input_path, "r", encoding="utf-8") as infile:
            raw_data = json.load(infile)
        
        # Ensure the data is an array
        if not isinstance(raw_data, list):
            raise ValueError("Input JSON must be an array of strings.")
        
        # Clean each entry
        cleaned_data = [clean_text(entry) for entry in raw_data if isinstance(entry, str)]
        
        # Save the cleaned dataset to a JSON file
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)
        
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the cleaning process
clean_dataset(input_file, output_file)
