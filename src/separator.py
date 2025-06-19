import json
import os
import sys
import random

def split_json(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        print(f"Failed to decode JSON file {json_file}. Trying alternative encoding...")
        with open(json_file, 'r', encoding='latin-1') as f:
            data = json.load(f)

    random.shuffle(data)  # Mezclar aleatoriamente los datos para hacer la división equitativa
    split_index = int(len(data) * 0.8)
    
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    base, ext = os.path.splitext(json_file)
    train_json_file = f"{base}_train{ext}"
    test_json_file = f"{base}_test{ext}"
    
    with open(train_json_file, 'w') as f:
        json.dump(train_data, f, indent=4)
        print(f"Train JSON file saved as {train_json_file}")
    
    with open(test_json_file, 'w') as f:
        json.dump(test_data, f, indent=4)
        print(f"Test JSON file saved as {test_json_file}")

def process_directory(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            json_file = os.path.join(directory, file_name)
            print(f"Processing {json_file}...")
            split_json(json_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    process_directory(directory)
