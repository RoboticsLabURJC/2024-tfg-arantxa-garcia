"""
Reduce the size of a JSON file by keeping only iterations divisible by a given number.

Usage:
    python script.py <json_file> <divisor>

Example:
    python script.py /path/to/file.json 2

The reduced JSON file will be saved with the same name as the original, 
but with a suffix indicating the divisor (e.g., file_2.json).
"""

import json
import os
import sys

def reduce_json(json_file, divisor):
    try:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        print(f"Failed to decode JSON file {json_file}. Trying alternative encoding...")
        with open(json_file, 'r', encoding='latin-1') as f:
            data = json.load(f)

    new_json = []
    counter = 0

    for iteration in data:  # Save only the iterations divisible by divisor
        if counter % divisor == 0:
            new_json.append(iteration)
        counter += 1

    base, ext = os.path.splitext(json_file)
    new_json_file = f"{base}_{divisor}{ext}"
    
    with open(new_json_file, 'w') as f:
        json.dump(new_json, f, indent=4)
        print(f"Reduced JSON file saved as {new_json_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <json_file> <divisor>")
        sys.exit(1)

    json_file = sys.argv[1]
    try:
        divisor = int(sys.argv[2])
    except ValueError:
        print("Divisor must be an integer.")
        sys.exit(1)

    reduce_json(json_file, divisor)
