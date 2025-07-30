import json
import argparse

def convert_json(input_path, output_path):
    # Read the input JSON file
    with open(input_path, 'r') as f:
        index_to_class = json.load(f)

    # Transform the data to the desired format
    name_to_image = {value: f"image refer {key}" for key, value in index_to_class.items()}

    # Write the output JSON file
    with open(output_path, 'w') as f:
        json.dump(name_to_image, f, indent=4)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert index_to_class JSON to name_to_image JSON')
    parser.add_argument('--input', default='./ckpts/index_to_class_mapping.json', help='Input JSON file path')
    parser.add_argument('--output', default='./tests/reference_image_data.json', help='Output JSON file path')
    args = parser.parse_args()

    # Run the conversion
    convert_json(args.input, args.output)