import json
import os
import sys
import numpy as np

def reduce_json(json_file, to_remove, mode="equally"):
    try:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        print(f"Failed to decode JSON file {json_file}. Trying alternative encoding...")
        with open(json_file, 'r', encoding='latin-1') as f:
            data = json.load(f)
    
    total_len = len(data)
    if to_remove >= total_len:
        print("to_remove is greater than or equal to the number of elements in the JSON. Returning an empty list.")
        new_json = []
    else:
        if mode == "end":
            new_json = data[:-to_remove] 
        else:  # Default to "equally"
            indices_to_remove = set(np.round(np.linspace(0, total_len - 1, to_remove, endpoint=False)).astype(int))
            new_json = [entry for idx, entry in enumerate(data) if idx not in indices_to_remove]
    
    base, ext = os.path.splitext(json_file)
    new_json_file = f"{base}_{to_remove}{ext}"
    
    with open(new_json_file, 'w', encoding='utf-8') as f:
        json.dump(new_json, f, indent=4)
        print(f"Reduced JSON file saved as {new_json_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python script.py <json_file> <to_remove> [mode]")
        sys.exit(1)

    json_file = sys.argv[1]
    try:
        to_remove = int(sys.argv[2])
    except ValueError:
        print("to_remove must be an integer.")
        sys.exit(1)
    
    mode = sys.argv[3] if len(sys.argv) == 4 else "equally"
    if mode not in ["equally", "end"]:
        print("Invalid mode. Use 'equally' or 'end'.")
        sys.exit(1)
    
    reduce_json(json_file, to_remove, mode)
