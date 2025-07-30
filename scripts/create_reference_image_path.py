import json
import argparse
import os

def convert_json(input_path, output_path, reference_dir):
    """
    Convert index_to_class JSON to name_to_image JSON with file paths, including all images in reference directory.
    
    Args:
        input_path (str): Path to input JSON file
        output_path (str): Path to output JSON file
        reference_dir (str): Directory containing reference images
    """
    # Define valid image extensions
    valid_extensions = ['jpg', 'png', 'jpeg']

    # Read the input JSON file
    try:
        with open(input_path, 'r') as f:
            index_to_class = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input JSON file {input_path} does not exist")
    except json.JSONDecodeError:
        raise ValueError(f"Input JSON file {input_path} is invalid")

    # Initialize dictionary for name to image path mapping
    name_to_image = {}

    # Process class names from input JSON
    json_entries = 0
    for key, value in index_to_class.items():
        if not isinstance(value, str):
            print(f"Warning: Skipping non-string class name {value} for key {key}")
            continue
        # Try each valid extension for the image file
        image_path = ""
        for ext in valid_extensions:
            potential_path = os.path.join(reference_dir, f"{value}.{ext.lower()}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        name_to_image[value] = image_path
        json_entries += 1

    # Scan reference directory for all images with valid extensions
    dir_entries = 0
    for filename in os.listdir(reference_dir):
        # Check if file has a valid image extension
        if any(filename.lower().endswith(f".{ext}") for ext in valid_extensions):
            # Extract the base name (without extension)
            base_name = os.path.splitext(filename)[0]
            # Only add to dictionary if not already present (avoid overwriting JSON-derived entries)
            if base_name not in name_to_image:
                image_path = os.path.join(reference_dir, filename)
                name_to_image[base_name] = image_path
                dir_entries += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the output JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(name_to_image, f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to write output JSON file {output_path}: {str(e)}")

    # Print notification
    total_entries = len(name_to_image)
    print(f"Successfully created reference image path dictionary with {total_entries} entries "
          f"({json_entries} from input JSON, {dir_entries} from reference directory). "
          f"Output saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert index_to_class JSON to name_to_image JSON with file paths, including all images in reference directory')
    parser.add_argument('--input', default='./ckpts/index_to_class_mapping.json', 
                        help='Input JSON file path')
    parser.add_argument('--output', default='./data/reference_data/reference_image_data.json', 
                        help='Output JSON file path')
    parser.add_argument('--reference-dir', default='data/reference_data/images', 
                        help='Directory containing reference images')

    args = parser.parse_args()

    # Ensure reference directory exists
    if not os.path.exists(args.reference_dir):
        raise FileNotFoundError(f"Reference directory {args.reference_dir} does not exist")

    # Run the conversion
    convert_json(args.input, args.output, args.reference_dir)