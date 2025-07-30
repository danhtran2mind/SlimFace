# SlimFace Inference Documentation

## Table of Contents
- [Introduction](#introduction)
- [Setup Guide](#setup-guide)
  - [System Requirements](#system-requirements)
  - [Directory Layout](#directory-layout)
  - [Installation Steps](#installation-steps)
- [Script Details](#script-details)
  - [inference.py: Face Classification](#inferencepy-face-classification)
    - [Command-Line Arguments](#command-line-arguments)
    - [Running the Script](#running-the-script)
  - [end2end_inference.py: Classification with Embedding Validation](#end2end_inferencepy-classification-with-embedding-validation)
    - [Command-Line Arguments](#command-line-arguments-1)
    - [Running the Script](#running-the-script-1)
- [Custom Usage Examples](#custom-usage-examples)
  - [Batch Classification with YOLO](#batch-classification-with-yolo)
  - [End-to-End Inference with Custom References](#end-to-end-inference-with-custom-references)
  - [Custom Model with Adjusted Resolution](#custom-model-with-adjusted-resolution)
- [Additional Notes](#additional-notes)

## Introduction
This document outlines the usage of `inference.py` and `end2end_inference.py` scripts from the SlimFace project for face classification and embedding validation. It provides setup instructions, detailed argument descriptions, and example usage scenarios.

## Setup Guide

### System Requirements
- **Python**: Version 3.10 or higher.
- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements/requirements_inference.txt
  ```
- **Model Files**: Ensure the trained model file (`SlimFace_efficientnet_b3_full_model.pth`) and index-to-class mapping JSON file (`index_to_class_mapping.json`) are in the `ckpts/` directory.
- **Reference Images** (for `end2end_inference.py`): A JSON file mapping class names to reference image paths (e.g., `reference_image_data.json`).
- **EdgeFace Model** (for `end2end_inference.py`): Ensure the EdgeFace model file (e.g., `edgeface_base.pt`) is in the `ckpts/idiap/` directory.

### Directory Layout
Ensure the following structure for model and data files:
```
SlimFace/
├── ckpts/
│   ├── SlimFace_efficientnet_b3_full_model.pth
│   ├── index_to_class_mapping.json
│   └── idiap/
│       └── edgeface_base.pt
├── data/
│   └── reference_data/
│       └── reference_image_data.json
├── tests/
│   └── test_images/
│       └── dont_know.jpg
├── src/
│   └── slimface/
│       └── inference/
│           ├── inference.py
│           └── end2end_inference.py
```

### Installation Steps
1. Install dependencies:
   ```bash
   pip install -r requirements/requirements_inference.txt
   ```
2. Set up third-party dependencies:
   ```bash
   python scripts/setup_third_party.py
   ```
3. Download model checkpoints:
   ```bash
   python scripts/download_ckpts.py
   ```

## Script Details

### inference.py: Face Classification
This script performs face classification on input images using a trained model. It preprocesses images, runs them through the model, and outputs predicted class names with confidence scores.

#### Command-Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_path` | `str` | **Required** | Path to a single image or a directory of images (`.jpg`, `.jpeg`, `.png`) for inference. |
| `--index_to_class_mapping_path` | `str` | `ckpts/index_to_class_mapping.json` | Path to the JSON file containing the index-to-class mapping. |
| `--model_path` | `str` | `ckpts/SlimFace_efficientnet_b3_full_model.pth` | Path to the trained model in TorchScript format (`.pth` file). |
| `--algorithm` | `str` | `yolo` | Face detection algorithm to use (`mtcnn` or `yolo`). |
| `--accelerator` | `str` | `auto` | Accelerator type for inference (`cpu`, `gpu`, or `auto`). |
| `--resolution` | `int` | `224` | Resolution for input images (e.g., 224 for 224x224). |

#### Running the Script
To classify a single image:
```bash
python src/slimface/inference/inference.py --input_path tests/test_images/dont_know.jpg
```

To classify all images in a directory:
```bash
python src/slimface/inference/inference.py --input_path tests/test_images/
```

**Output Example**:
```
Image: tests/test_images/dont_know.jpg
Predicted Class: Robert Downey Jr
Confidence: 0.9293
```

### end2end_inference.py: Classification with Embedding Validation
This script extends `inference.py` by adding EdgeFace embedding validation. It classifies an image and compares its embedding with a reference image for the predicted class to confirm the prediction based on a similarity threshold.

#### Command-Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--unknown_image_path` | `str` | **Required** | Path to a single image or a directory of images (`.jpg`, `.jpeg`, `.png`). |
| `--reference_dict_path` | `str` | `data/reference_data/reference_image_data.json` | Path to the JSON file mapping class names to reference image paths. |
| `--index_to_class_mapping_path` | `str` | `ckpts/index_to_class_mapping.json` | Path to the index-to-class mapping JSON file. |
| `--model_path` | `str` | `ckpts/SlimFace_efficientnet_b3_full_model.pth` | Path to the classifier model (`.pth` file). |
| `--edgeface_model_path` | `str` | `ckpts/idiap/edgeface_base.pt` | Path to the EdgeFace model file. |
| `--algorithm` | `str` | `yolo` | Face detection algorithm (`mtcnn` or `yolo`). |
| `--accelerator` | `str` | `auto` | Accelerator type (`cpu`, `gpu`, or `auto`). |
| `--resolution` | `int` | `224` | Input image resolution (e.g., 224 for 224x224). |
| `--similarity_threshold` | `float` | `0.3` | Cosine similarity threshold for confirming predictions. |

#### Running the Script
To classify and validate a single image:
```bash
python src/slimface/inference/end2end_inference.py \
    --unknown_image_path tests/test_images/dont_know.jpg \
    --reference_dict_path data/reference_data/reference_image_data.json \
    --similarity_threshold 0.6
```

**Output Example**:
```
Image: tests/test_images/dont_know.jpg, Predicted Class: Robert Downey Jr, Confidence: 0.9293, Similarity: 0.6033, Confirmed: True
```

## Custom Usage Examples

### Batch Classification with YOLO
To classify all images in a directory using the YOLO algorithm and automatic accelerator selection:
```bash
python src/slimface/inference/inference.py \
    --input_path tests/test_images/ \
    --algorithm yolo \
    --accelerator auto \
    --model_path ckpts/SlimFace_efficientnet_b3_full_model.pth \
    --index_to_class_mapping_path ckpts/index_to_class_mapping.json \
    --resolution 224
```

### End-to-End Inference with Custom References
To perform classification and embedding validation with a custom reference image dataset and a higher similarity threshold:
```bash
python src/slimface/inference/end2end_inference.py \
    --unknown_image_path tests/test_images/dont_know.jpg \
    --reference_dict_path custom_references.json \
    --index_to_class_mapping_path ckpts/index_to_class_mapping.json \
    --model_path ckpts/SlimFace_efficientnet_b3_full_model.pth \
    --edgeface_model_path ckpts/idiap/edgeface_base.pt \
    --algorithm yolo \
    --accelerator auto \
    --resolution 224 \
    --similarity_threshold 0.7
```

**Example `custom_references.json`**:
```json
{
    "Robert Downey Jr": "data/references/rdj_reference.jpg",
    "Scarlett Johansson": "data/references/scarlett_reference.jpg"
}
```

### Custom Model with Adjusted Resolution
To use a custom-trained model and a different resolution (e.g., 112x112):
```bash
python src/slimface/inference/inference.py \
    --input_path tests/test_images/ \
    --model_path ckpts/custom_model.pth \
    --index_to_class_mapping_path ckpts/custom_mapping.json \
    --resolution 112 \
    --algorithm yolo \
    --accelerator cpu
```

## Additional Notes
- **Error Handling**: Both scripts handle cases where face detection fails by falling back to resizing the original image.
- **Performance**: Use GPU acceleration (`--accelerator gpu`) for faster inference if a compatible GPU is available.
- **Reference Images**: For `end2end_inference.py`, ensure the `reference_dict_path` JSON file contains valid paths to reference images for each class.
- **Model Compatibility**: The scripts assume the model is in TorchScript format (`.pth`). Ensure the model and mapping files are compatible with the trained dataset.

For further assistance, refer to the SlimFace repository or contact the project maintainers.