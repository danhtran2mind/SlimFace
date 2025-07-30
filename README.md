# SlimFace: Efficient Face Recognition 🚀

[![GitHub Stars](https://img.shields.io/github/stars/danhtran2mind/SlimFace?style=social&label=Repo%20Stars)](https://github.com/danhtran2mind/SlimFace/stargazers)
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fdanhtran2mind%2FSlimFace&label=Repo+Views&icon=github&color=%236f42c1&message=&style=social&tz=UTC)

[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-blue.svg?logo=huggingface)](https://huggingface.co/docs/hub)
[![accelerate](https://img.shields.io/badge/accelerate-blue.svg?logo=pytorch)](https://huggingface.co/docs/accelerate)
[![bitsandbytes](https://img.shields.io/badge/bitsandbytes-blue.svg)](https://github.com/TimDettmers/bitsandbytes)
[![torch](https://img.shields.io/badge/torch-blue.svg?logo=pytorch)](https://pytorch.org/)
[![Pillow](https://img.shields.io/badge/Pillow-blue.svg)](https://pypi.org/project/pillow/)
[![numpy](https://img.shields.io/badge/numpy-blue.svg?logo=numpy)](https://numpy.org/)
[![transformers](https://img.shields.io/badge/transformers-blue.svg?logo=huggingface)](https://huggingface.co/docs/transformers)
[![torchvision](https://img.shields.io/badge/torchvision-blue.svg?logo=pytorch)](https://pytorch.org/vision/stable/index.html)
[![diffusers](https://img.shields.io/badge/diffusers-blue.svg?logo=huggingface)](https://huggingface.co/docs/diffusers)
[![gradio](https://img.shields.io/badge/gradio-blue.svg?logo=gradio)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Troubleshooting OpenCV](#troubleshooting-opencv)
  - [Third-Party Dependencies](#third-party-dependencies)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Reference Data Setup](#reference-data-setup)
  - [Base Model](#base-model)
  - [Training](#training)
  - [Training Hyperparameters](#training-hyperparameters)
- [Demonstration](#demonstration)
  - [Interactive Demo](#interactive-demo)
  - [Run Locally](#run-locally)
- [Metrics](#metrics)
- [Environment](#environment)
- [Credits and Citation](#credits-and-citation)
- [Contact](#contact)

## Introduction

SlimFace is a robust and efficient framework for face recognition, designed to enable developers to build high-accuracy models using transfer learning with pre-trained TorchVision architectures. Optimized for edge devices, SlimFace delivers a scalable solution with minimal computational overhead, supporting custom datasets for tailored applications and providing an interactive interface for seamless inference and testing.

## Key Features

- **High Accuracy**: Utilizes pre-trained TorchVision models fine-tuned for precise face classification.
- **Efficient Pipeline**: Optimized for edge devices to ensure low computational requirements.
- **Flexible Training**: Supports custom datasets, enabling tailored face recognition solutions.
- **Gradio Demo**: Provides an interactive interface for easy model inference and testing.

## Installation

### Step 1: Clone the Repository

To begin, clone the SlimFace repository to your local machine:

```bash
git clone https://github.com/danhtran2mind/SlimFace
cd SlimFace
```

### Step 2: Install Dependencies

Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements/requirements.txt
```

For specific use cases, alternative dependency configurations are available:

- **Compatible Dependencies**:
  ```bash
  pip install -r requirements/requirements_compatible.txt
  ```
- **End-to-End Inference**:
  ```bash
  pip install -r requirements/requirements_inference.txt
  ```

### Troubleshooting OpenCV

If you encounter issues with OpenCV, ensure the following dependencies are installed:

```bash
sudo apt update
sudo apt install -y libglib2.0-0 libgl1-mesa-dev
```

### Third-Party Dependencies

Set up third-party dependencies by running:

```bash
python scripts/setup_third_party.py
```

## Usage

### Dataset Preparation

Prepare your dataset by following the [Data Processing Guide](./docs/data_processing.md). For a quick setup, execute:

```bash
python scripts/process_dataset.py
```

### Reference Data Setup

1. Place reference images in `data/reference_data/images` and ensure they are mapped to `index_to_class_mapping.json`.
2. Generate the reference dictionary with:

```bash
python scripts/create_reference_image_path.py
```

For custom paths, use:

```bash
python scripts/create_reference_image_path.py \
    --input <path_to_index_to_class_mapping.json> \
    --output <path_to_reference_image_data.json>
```

### Base Model

SlimFace leverages pre-trained TorchVision models fine-tuned for face recognition. Download model checkpoints using:

```bash
python scripts/download_ckpts.py
```

## Training

1. Configure Accelerate for distributed training:

```bash
accelerate config default
```

2. Launch the training process:

```bash
accelerate launch src/slimface/training/accelerate_train.py
```

For detailed instructions, refer to the [Training Documentation](./docs/training/training_docs.md).

### Training Hyperparameters

Default hyperparameters are optimized for performance and can be customized in `src/slimface/training/accelerate_train.py`. Refer to the [Training Documentation](./docs/training/training_docs.md) for guidance.

## Inference

### Class Name Inference
To classify an image and retrieve its class name, run:
```bash
python src/slimface/inference/inference.py --input_path <path_to_image>
```

### Class Name and Image Embedding Comparison
To perform class name inference and compare image embeddings, use:
```bash
python src/slimface/inference/end2end_inference.py \
    --unknown_image_path tests/test_images/dont_know.jpg \
    --similarity_threshold 0.3
```

### Additional Arguments
For more details and available options, refer to the [Inference Documentation](docs/inference/inference_doc.md).

## Demonstration

### Interactive Demo

Explore the interactive demo hosted on HuggingFace:

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace_Space-blue.svg?logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/SlimFace-demo)

Below is a screenshot of the SlimFace Demo GUI:

<img src="./assets/gradio_app_demo.jpg" alt="SlimFace Demo" height="600">

### Run Locally

To run the Gradio application locally at the default address `localhost:7860`, execute:

```bash
python apps/gradio_app.py
```

## Metrics

Evaluate model performance using standard metrics such as accuracy and F1-score. Detailed evaluation procedures are available in the [Training Documentation](./docs/training/training_docs.md).

## Environment

SlimFace requires the following environment:

- **Python**: 3.10 or higher
- **Key Libraries**: Refer to [Requirements Compatible](./requirements/requirements_compatible.txt) for compatible dependencies.

## Credits and Citation

SlimFace builds upon the [otroshi/edgeface](https://github.com/otroshi/edgeface) project by [Hatef Otroshi](https://github.com/otroshi), with enhancements and bug fixes available at [danhtran2mind/edgeface](https://github.com/danhtran2mind/edgeface/tree/main/face_alignment).

## Contact

For questions, issues, or support, please use the [GitHub Issues](https://github.com/danhtran2mind/SlimFace/issues) tab or contact the maintainer via the [HuggingFace Community](https://huggingface.co/spaces/danhtran2mind/SlimFace-demo/discussions).