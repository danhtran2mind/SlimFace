# SlimFace: Efficient Face Recognition 🚀

[![GitHub Stars](https://img.shields.io/github/stars/danhtran2mind/SlimFace?style=social&label=Repo%20Stars)](https://github.com/danhtran2mind/SlimFace/stargazers)
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fdanhtran2mind%2FSlimFace&label=Repo+Views&icon=github&color=%236f42c1&message=&style=social&tz=UTC)

[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-blue.svg?logo=huggingface)](https://huggingface.co/docs/hub)
[![accelerate](https://img.shields.io/badge/accelerate-blue.svg?logo=pytorch)](https://huggingface.co/docs/accelerate)
[![torch](https://img.shields.io/badge/torch-blue.svg?logo=pytorch)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/transformers-blue.svg?logo=huggingface)](https://huggingface.co/docs/transformers)
[![torchvision](https://img.shields.io/badge/torchvision-blue.svg?logo=pytorch)](https://pytorch.org/vision/stable/index.html)
[![gradio](https://img.shields.io/badge/gradio-blue.svg?logo=gradio)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## Credits and Citation
>
> ℹ️ This project leverages code from [![Built on edgeface](https://img.shields.io/badge/Built%20on-otroshi%2Fedgeface-blue?style=flat&logo=github)](https://github.com/otroshi/edgeface) by [![Hatef Otroshi](https://img.shields.io/badge/GitHub-Hatef_Otroshi-blue?style=flat&logo=github)](https://github.com/otroshi), with our own bug fixes and enhancements available at [![Edgeface Enhancements](https://img.shields.io/badge/GitHub-danhtran2mind%2Fedgeface-blue?style=flat&logo=github)](https://github.com/danhtran2mind/edgeface/tree/main/face_alignment). Explore additional EdgeFace models on the Hugging Face Hub: [![Idiap HuggingFace](https://img.shields.io/badge/HuggingFace-Idiap-yellow?style=flat&logo=huggingface)](https://huggingface.co/Idiap).
>
> If this project is helpful for your research, please consider citing the original paper:
>
> **Edgeface: Efficient face recognition model for edge devices**  
> *George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien*  
> *IEEE Transactions on Biometrics, Behavior, and Identity Science (2024)*
>
> **If you use this work in your research, please cite the original paper:**
> ```bibtex
> @article{edgeface,
>   title={Edgeface: Efficient face recognition model for edge devices},
>   author={George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien},
>   journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
>   year={2024}
> }
> ```

## Introduction

SlimFace is a robust and efficient framework for face recognition, designed to enable developers to build high-accuracy models using transfer learning with pre-trained TorchVision architectures. Optimized for edge devices, SlimFace delivers a scalable solution with minimal computational overhead, supporting custom datasets for tailored applications and providing an interactive interface for seamless inference and testing.

## Key Features

- **High Accuracy**: Utilizes pre-trained TorchVision models fine-tuned for precise face classification.
- **Efficient Pipeline**: Optimized for edge devices to ensure low computational requirements.
- **Flexible Training**: Supports custom datasets, enabling tailored face recognition solutions.
- **Gradio Demo**: Provides an interactive interface for easy model inference and testing.

## Datasets
You can explore more in this HuggingFace Dataset available at the given link for further details: [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-vasukipatel%2Fface--recognition--dataset-blue?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset).

The dataset preparation script is located at `scripts/process_dataset.py`.
## Base Model

SlimFace leverages pre-trained TorchVision models fine-tuned for face recognition. Follow this `scripts/download_ckpts.py` to download model checkpoints.

For more details on Base Models, refer to the PyTorch Pretrained Model Documentation: [![PyTorch Documentation](https://img.shields.io/badge/PyTorch-Pretrain%20Model%20Docs-orange?style=flat&logo=pytorch)](https://docs.pytorch.org/vision/main/models.html).

## Demonstration

### Interactive Demo

Explore the interactive demo hosted on HuggingFace:
[![HuggingFace Space Demo](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FSlimFace--demo-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/SlimFace-demo)

Below is a screenshot of the SlimFace Demo GUI:

<img src="./assets/gradio_app_demo.jpg" alt="SlimFace Demo" height="600">

### Run Locally

To run the Gradio application locally at the default address `localhost:7860`, execute:

```bash
python apps/gradio_app.py
```

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


## Training

1. Configure Accelerate for distributed training:

```bash
accelerate config default
```

2. Launch the training process:

```bash
accelerate launch src/slimface/training/accelerate_train.py
```

For detailed instructions, refer to the [Training Documentation](./docs/training/training_doc.md).

### Training Hyperparameters

Default hyperparameters are optimized for performance and can be customized in `src/slimface/training/accelerate_train.py`. Refer to the [Training Documentation](./docs/training/training_doc.md) for guidance.

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

## Metrics
Model Evaluation Results: Table summarizing train/validation loss and accuracy for four models.
For details you can see at [Models Metric](./docs/evaluation/comparision.md).




## Environment

SlimFace requires the following environment:

- **Python**: 3.10 or higher
- **Key Libraries**: Refer to [Requirements Compatible](./requirements/requirements_compatible.txt) for compatible dependencies.

## More Knowledge

For additional information on models and pre-trained weights, please refer to the following resources: [![PyTorch Documentation](https://img.shields.io/badge/PyTorch-Pretrain%20Model%20Docs-orange?style=flat&logo=pytorch)](https://docs.pytorch.org/vision/main/models.html)

The Documentation includes the `torchvision.models` subpackage, which offers pre-trained models for various image tasks. It provides metrics like Acc@1, Acc@5, Params, and GFLOPS for model evaluation.

## Contact

For questions, issues, or support, please use the [GitHub Issues](https://github.com/danhtran2mind/SlimFace/issues) tab or contact the maintainer via the [HuggingFace Community](https://huggingface.co/spaces/danhtran2mind/SlimFace-demo/discussions).
