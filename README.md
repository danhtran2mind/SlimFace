# SlimFace: Slim Face Recognition

> ## Credits and Citation
>
> ℹ️ This project leverages code from [![Built on edgeface](https://img.shields.io/badge/Built%20on-otroshi%2Fedgeface-blue?style=flat&logo=github)](https://github.com/otroshi/edgeface) by [![Hatef Otroshi](https://img.shields.io/badge/GitHub-Hatef_Otroshi-blue?style=flat&logo=github)](https://github.com/otroshi), with our own bug fixes and enhancements available at [![Edgeface Enhancements](https://img.shields.io/badge/GitHub-danhtran2mind%2Fedgeface-blue?style=flat&logo=github)](https://github.com/danhtran2mind/edgeface/tree/main/face_alignment).
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
git clone https://github.com/danhtran2mind/SlimFace

# Navigate into the newly created 'slimface' directory.
cd SlimFace
```
### Install Dependencies
**If Open-CV (CV2) does not work, run below CLI**
```bash
sudo apt update
sudo apt install -y libglib2.0-0
sudo apt install -y libgl1-mesa-dev
```
### Default install Dependencies
```bash
pip install -r requirements/requirements.txt
```
### Other install Dependencies
- For My Compatible
```bash
pip install -r requirements/requirements_compatible.txt
```
- For `End2end Inference`
```bash
pip install -r requirements/requirements_inference.txt
```
### Download Model Checkpoints
```bash
python scripts/download_ckpts.py
```
### Setup Third Party
```bash
python scripts/setup_third_party.py
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
accelerate launch src/slimface/training/accelerate_train.py
```

For additional help, you can refer to the [Training Documentation](./docs/training/training_docs.md) for more details.

### Inference
#### Create Reference Images Data at `data/reference_data/images`
For each class, you store an image in `data/reference_data/images` folder which are maped with `index_to_class_mapping.json`.

The structure like:
```markdown
data/reference_data/images/
├── 'Robert Downey Jr.jpg'
├── 'Tom Cruise.jpg'
└── ...
```


### Create Reference Dictionary from `index_to_class_mapping.json`

#### Steps
1. Place `index_to_class_mapping.json` in the `ckpts` folder.
2. Ensure reference images are in `data/reference_data/images`. Missing images will be set to `""` in `reference_image_data.json` (default in `data/reference_data` folder).
3. Run one of the following commands:

#### Commands
- **Default** (Output: `data/reference_data/reference_image_data.json`):
  ```bash
  python scripts/create_reference_image_path.py
  ```
- **Custom Paths**:
  ```bash
  python scripts/create_reference_image_path.py \
      --input <path_to_index_to_class_mapping.json> \
      --output <path_to_tests/reference_image_data.json>
  ```

#### Manual Option
Edit `reference_image_data.json` directly to add image paths as dictionary values.

## Demostration
```bash
python apps/gradio_app.py
```

https://huggingface.co/spaces/danhtran2mind/SlimFace-demo

## Project Description

This repository leverages code [![GitHub Repo](https://img.shields.io/badge/GitHub-danhtran2mind%2Fedgeface-blue?style=flat)](https://github.com/danhtran2mind/edgeface), a fork of [![GitHub Repo](https://img.shields.io/badge/GitHub-otroshi%2Fedgeface-blue?style=flat)](https://github.com/otroshi/edgeface), with numerous bug fixes and rewritten code for improved performance and stability.
