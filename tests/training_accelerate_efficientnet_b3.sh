#!/bin/bash

# Exit on any error
set -e

# Function to log messages
log_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        log_message "ERROR: Directory $1 does not exist"
        exit 1
    fi
}

# Function to check Python and required tools
check_requirements() {
    log_message "Checking requirements..."

    # Check for Python
    if ! command_exists python3; then
        log_message "ERROR: Python3 is not installed"
        exit 1
    fi

    # Check for accelerate
    if ! command_exists accelerate; then
        log_message "ERROR: Accelerate is not installed. Please install it using 'pip install accelerate'"
        exit 1
    fi
}

# Main script execution
main() {
    log_message "Starting training pipeline..."

    # Set variables
    DATASET_DIR="./data/processed_ds"
    SCRIPT_DIR="scripts"
    SRC_DIR="src/slim_face/training"

    # Check if required directories exist
    check_directory "$DATASET_DIR"
    check_directory "$SCRIPT_DIR"
    check_directory "$SRC_DIR"

    # Check requirements
    check_requirements

    # Process dataset
    log_message "Processing dataset..."
    python3 "${SCRIPT_DIR}/process_dataset.py" \
        --random_state 42 \
        --test_split_rate 0.2 \
        --augment || {
        log_message "ERROR: Dataset processing failed"
        exit 1
    }

    # Configure accelerate
    log_message "Configuring accelerate..."
    accelerate config default || {
        log_message "ERROR: Accelerate configuration failed"
        exit 1
    }

    # Launch training
    log_message "Starting model training..."
    accelerate launch "${SRC_DIR}/accelerate_train.py.py" \
        --batch_size 32 \
        --algorithm yolo \
        --learning_rate 1e-4 \
        --max_lr_factor 4 \
        --warmup_steps 0.05 \
        --num_epochs 100 \
        --dataset_dir "$DATASET_DIR" \
        --classification_model_name efficientnet_b3 || {
        log_message "ERROR: Training failed"
        exit 1
    }

    log_message "Training pipeline completed successfully"
}

# Trap Ctrl+C and exit gracefully
trap 'log_message "Script interrupted by user"; exit 1' INT

# Execute main function
main "$@"