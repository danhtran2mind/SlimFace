# Slim Face Recognition

> ## Credits and Citation
>
> ℹ️ This project is based on the [![Built on edgeface](https://img.shields.io/badge/Built%20on-otroshi%2Fedgeface-blue?style=flat&logo=github)](https://github.com/otroshi/edgeface) by [![Hatef Otroshi](https://img.shields.io/badge/GitHub-Hatef_Otroshi-blue?style=flat&logo=github)](https://github.com/otroshi), and includes our own bug fixes and enhancements.
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
## Usage
### Clone Repositories
```bash
# Clone the repository
git clone https://github.com/danhtran2mind/slim-face

# Navigate into the newly created 'slim-face' directory.
cd slim-face
```
### Install Dependencies
**If Open-CV (CV2) does not work, run below CLI**
```bash
sudo apt update
sudo apt install -y libglib2.0-0
sudo apt install -y libgl1-mesa-dev
```
```bash
pip install -r requirements.txt
```


## Data Preparation

## Pre-trained Model preparation
For detailed instructions on how to process and manage your data effectively, refer to the [Full guide for data processing](./docs/data_processing.md).

This is fast usage for dataset preparation
```bash
python scripts/process_dataset.py
```
## Training

1. Configure the default settings for Accelerate:
```bash
accelerate config default
```

2. Launch the training script using Accelerate:
```bash
accelerate launch src/slim_face/training/accelerate_train.py
```

For additional help, you can refer to the [Training Documentation](./docs/training/training_docs.md) for more details.

## Project Description

This repository is trained from [![GitHub Repo](https://img.shields.io/badge/GitHub-danhtran2mind%2Fedgeface-blue?style=flat)](https://github.com/danhtran2mind/edgeface), a fork of [![GitHub Repo](https://img.shields.io/badge/GitHub-otroshi%2Fedgeface-blue?style=flat)](https://github.com/otroshi/edgeface), with numerous bug fixes and rewritten code for improved performance and stability.
