import json
import argparse
import os

def concat_json_from_dir(directory, output_file):
    combined_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    combined_data.extend(data)
                elif isinstance(data, dict):
                    combined_data.append(data)
                else:
                    raise ValueError(f"El archivo {filename} no tiene una estructura compatible")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(combined_data, out, indent=4, ensure_ascii=False)
    
    print(f"Archivo combinado guardado en {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatena todos los archivos JSON en un directorio.")
    parser.add_argument("directory", help="Directorio con los archivos JSON")
    parser.add_argument("output", help="Archivo JSON de salida")
    
    args = parser.parse_args()
    concat_json_from_dir(args.directory, args.output)
