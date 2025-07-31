
# Training Documentation

This document outlines the command-line arguments and a concise overview of the training pipeline for a face classification model using PyTorch Lightning.   

# Training Arguments Documentation

This document outlines the command-line arguments and a concise overview of the training pipeline for a face classification model using PyTorch Lightning.

## Table of Contents

- [Arguments Table](#arguments-table)
- [Training Pipeline Overview](#training-pipeline-overview)
- [Supported Training Architectures](#supported-training-architectures)

## Arguments Table

| Argument Name                          | Type  | Description                                                                                                                    |
|----------------------------------------|-------|-------------------------------------------------------------------------------------------------------------------------------|
| `dataset_dir`                          | `str` | Path to the dataset directory containing `train_data` and `val_data` subdirectories with preprocessed face images organized by person. |
| `image_classification_models_config_path` | `str` | Path to the YAML configuration file defining model configurations, including model function, resolution, and weights.              |
| `batch_size`                           | `int` | Batch size for training and validation data loaders. Affects memory usage and training speed.                                  |
| `num_epochs`                           | `int` | Number of epochs for training the model. An epoch is one full pass through the training dataset.                               |
| `learning_rate`                        | `float` | Initial learning rate for the Adam optimizer used during training.                                                             |
| `max_lr_factor`                        | `float` | Multiplies the initial learning rate to determine the maximum learning rate during the warmup phase of the scheduler.           |
| `accelerator`                          | `str` | Type of accelerator for training. Options: `cpu`, `gpu`, `tpu`, `auto`. `auto` selects the best available device.              |
| `devices`                              | `int` | Number of devices (e.g., GPUs) to use for training. Relevant for multi-GPU training.                                           |
| `algorithm`                            | `str` | Face detection algorithm for preprocessing images. Options: `mtcnn`, `yolo`.                                                  |
| `warmup_steps`                         | `float` | Fraction of total training steps for the warmup phase of the learning rate scheduler (e.g., `0.05` means 5% of total steps).  |
| `total_steps`                          | `int` | Total number of training steps. If `0`, calculated as epochs × steps per epoch (based on dataset size and batch size).         |
| `classification_model_name`             | `str` | Name of the classification model to use, as defined in the YAML configuration file.                                            |

## Training Pipeline Overview

The training pipeline preprocesses face images, fine-tunes a classification head on a pretrained model, and trains using PyTorch Lightning. Key components:

1. **Preprocessing**: Aligns faces using `yolo` or `mtcnn`, caches resized images (`preprocess_and_cache_images`).
2. **Dataset**: `FaceDataset` loads pre-aligned images, applies normalization, and assigns labels by person.
3. **Model**: `FaceClassifier` pairs a frozen pretrained model (e.g., EfficientNet) with a custom classification head.
4. **Training**: `FaceClassifierLightning` manages training with Adam optimizer, cosine annealing scheduler, and logs loss/accuracy.
5. **Configuration**: Loads model details from YAML (`load_model_configs`), uses `DataLoader` with multiprocessing, and saves models via `CustomModelCheckpoint`.
6. **Execution**: `main` orchestrates preprocessing, data loading, model training, and saves full model and classifier head.

## Supported Training Architectures
- **EfficientNet**: B0-B7 and V2 (Small, Medium, Large) for balanced performance and efficiency. 📸  
- **RegNet**: X/Y series (400MF to 128GF) for optimized computation across diverse hardware. 💻  
- **Vision Transformers (ViT)**: B_16, B_32, H_14, L_16, L_32 for cutting-edge feature extraction. 🚀  