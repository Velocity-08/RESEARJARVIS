import json

# Define file paths
input_file = "cleaned_dataset.json"  # Input: Cleaned dataset
output_file = "filtered_dataset.json"  # Output: Filtered dataset

# Function to filter dataset
def filter_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        cleaned_data = json.load(infile)

    filtered_data = []
    seen_entries = set()  # To track duplicates

    for entry in cleaned_data:
        # Check for valid 'query' and 'response'
        if "query" in entry and "response" in entry:
            query = entry["query"].strip()
            response = entry["response"].strip()

            # Skip if either field is empty
            if not query or not response:
                continue

            # Combine query and response to check for duplicates
            combined = f"{query} | {response}"
            if combined in seen_entries:
                continue  # Skip duplicates
            
            # Add to filtered data and mark as seen
            filtered_data.append({"query": query, "response": response})
            seen_entries.add(combined)

    # Save filtered data to output JSON file
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(filtered_data, outfile, indent=4)

    print(f"Filtered data saved to {output_path}")

# Run the filtering process
filter_dataset(input_file, output_file)
