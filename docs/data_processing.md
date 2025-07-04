# Data Processing for Slim-Face Training 🖼️

## Table of Contents

- [Data Processing for Slim-Face Training 🖼️](#data-processing-for-slim-face-training-)
  - [Command-Line Arguments](#command-line-arguments)
    - [Command-Line Arguments for `process_dataset.py`](#command-line-arguments-for-process_datasetpy)
    - [Example Usage](#example-usage)
  - [Step-by-step process for handling a dataset](#step-by-step-process-for-handling-a-dataset)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Process the Dataset](#step-2-process-the-dataset)
      - [Option 1: Using Dataset from Kaggle](#option-1-using-dataset-from-kaggle)
      - [Option 2: Using a Custom Dataset](#option-2-using-a-custom-dataset)

## Command-Line Arguments
### Command-Line Arguments for `process_dataset.py`
When running `python scripts/process_dataset.py`, you can customize the dataset processing with the following command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_slug` | `str` | `vasukipatel/face-recognition-dataset` | The Kaggle dataset slug in `username/dataset-name` format. Specifies which dataset to download from Kaggle. |
| `--base_dir` | `str` | `./data` | The base directory where the dataset will be stored and processed. |
| `--augment` | `flag` | `False` | Enables data augmentation (e.g., flipping, rotation) for training images to increase dataset variety. Use `--augment` to enable. |
| `--random_state` | `int` | `42` | Random seed for reproducibility in the train-test split. Ensures consistent splitting across runs. |
| `--test_split_rate` | `float` | `0.2` | Proportion of data to use for validation (between 0 and 1). For example, `0.2` means 20% of the data is used for validation. |
| `--rotation_range` | `int` | `15` | Maximum rotation angle in degrees for data augmentation (if `--augment` is enabled). Images may be rotated randomly within this range. |
| `--source_subdir` | `str` | `Original Images/Original Images` | Subdirectory within `raw_dir` containing the images to process. Used for both Kaggle and custom datasets. |
| `--delete_raw` | `flag` | `False` | Deletes the raw folder after processing to save storage. Use `--delete_raw` to enable. |

### Example Usage
To process a Kaggle dataset with augmentation and a custom validation split:

```bash
python scripts/process_dataset.py \
    --augment \
    --test_split_rate 0.3 \
    --rotation_range 15
```

To process a **custom dataset** with a specific subdirectory and delete the raw folder:

```bash
python scripts/process_dataset.py \
    --source_subdir your_custom_dataset_dir \
    --delete_raw
```
## Step-by-step process for handling a dataset
These options allow flexible dataset processing tailored to your needs. 🚀

### Step 1: Clone the Repository
Ensure the `slim-face` project is set up by cloning the repository and navigating to the project directory:

```bash
git clone https://github.com/danhtran2mind/slim-face/
cd slim-face
```

### Step 2: Process the Dataset

#### Option 1: Using Dataset from Kaggle
To download and process the sample dataset from Kaggle, run:

```bash
python scripts/process_dataset.py
```

This script organizes the dataset into the following structure under `data/`:

```markdown
data/
├── processed_ds/
│   ├── train_data_aligned_efficientnet_b0/
│   │   ├── Charlize Theron/
│   │   │   ├── Charlize Theron_70.jpg
│   │   │   ├── Charlize Theron_46.jpg
│   │   │   ...
│   │   ├── Dwayne Johnson/
│   │   │   ├── Dwayne Johnson_58.jpg
│   │   │   ├── Dwayne Johnson_9.jpg
│   │   │   ...
│   ├── train_data/
│   │   ├── Charlize Theron/
│   │   │   ├── Charlize Theron_70.jpg
│   │   │   ├── Charlize Theron_46.jpg
│   │   │   ...
│   │   ├── Dwayne Johnson/
│   │   │   ├── Dwayne Johnson_58.jpg
│   │   │   ├── Dwayne Johnson_9.jpg
│   │   │   ...
│   └── val_data/
│       ├── Charlize Theron/
│       │   ├── Charlize Theron_60.jpg
│       │   ├── Charlize Theron_45.jpg
│       │   ...
│       ├── Dwayne Johnson/
│       │   ├── Dwayne Johnson_11.jpg
│       │   ├── Dwayne Johnson_46.jpg
│       │   ...
├── raw/
│   ├── Dataset.csv
│   ├── Faces/
│   │   ├── Jessica Alba_90.jpg
│   │   ├── Hugh Jackman_70.jpg
│   │   ...
│   ├── dataset.zip
│   ├── Original Images/
│   │   ├── Charlize Theron/
│   │   │   ├── Charlize Theron_60.jpg
│   │   │   ├── Charlize Theron_70.jpg
│   │   │   ...
│   │   ├── Dwayne Johnson/
│   │   │   ├── Dwayne Johnson_11.jpg
│   │   │   ├── Dwayne Johnson_58.jpg
│   │   │   ...
└── .gitignore
```

#### Option 2: Using a Custom Dataset
If you prefer to use your own dataset, place it in `./data/raw/your_custom_dataset_dir/` with the following structure:

```markdown
data/
├── raw/
│   ├── your_custom_dataset_dir/
│   │   ├── Charlize Theron/
│   │   │   ├── Charlize Theron_60.jpg
│   │   │   ├── Charlize Theron_70.jpg
│   │   │   ...
│   │   ├── Dwayne Johnson/
│   │   │   ├── Dwayne Johnson_11.jpg
│   │   │   ├── Dwayne Johnson_58.jpg
│   │   │   ...
```

If you use your dataset, you do not need to include only human faces, because **we support face extraction using face detection**, and all extracted faces are saved at `data/processed_ds`.

Then, process your custom dataset by specifying the subdirectory:

```bash
python scripts/process_dataset.py \
    --source_subdir your_custom_dataset_dir
```

This ensures your dataset is properly formatted for training. 🚀