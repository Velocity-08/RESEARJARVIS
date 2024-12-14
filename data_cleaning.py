import re

# Define file paths
input_file = "raw_dataset.json"  # Input: Raw dataset
output_file = "cleaned_dataset.json"  # Output: Cleaned dataset

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
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            cleaned_line = clean_text(line)
            if cleaned_line:  # Skip empty lines
                outfile.write(cleaned_line + "\n")
    print(f"Cleaned data saved to {output_path}")

# Run the cleaning process
clean_dataset(input_file, output_file)
