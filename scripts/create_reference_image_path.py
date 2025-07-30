import json
import argparse
import os

def convert_json(input_path, output_path, reference_dir, image_extension):
    # Read the input JSON file
    with open(input_path, 'r') as f:
        index_to_class = json.load(f)

    # Transform the data to use image paths or empty string if path doesn't exist
    name_to_image = {}
    for key, value in index_to_class.items():
        image_path = os.path.join(reference_dir, f"{value}.{image_extension}")
        name_to_image[value] = image_path if os.path.exists(image_path) else ""

    # Write the output JSON file
    with open(output_path, 'w') as f:
        json.dump(name_to_image, f, indent=4)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert index_to_class JSON to name_to_image JSON with file paths')
    parser.add_argument('--input', default='./ckpts/index_to_class_mapping.json', 
                        help='Input JSON file path')
    parser.add_argument('--output', default='./data/reference_image_data.json', 
                        help='Output JSON file path')
    parser.add_argument('--reference-dir', default='data/reference_data/images', 
                        help='Directory containing reference images')
    parser.add_argument('--image-extension', default='jpg', 
                        help='Image file extension (jpg, png, etc.)')

    args = parser.parse_args()

    # Ensure reference directory exists
    if not os.path.exists(args.reference_dir):
        raise FileNotFoundError(f"Reference directory {args.reference_dir} does not exist")

    # Run the conversion
    convert_json(args.input, args.output, args.reference_dir, args.image_extension)