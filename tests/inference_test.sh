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

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        log_message "ERROR: File $1 does not exist"
        exit 1
    fi
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
}

# Main script execution
main() {
    log_message "Starting inference pipeline..."

    # Set variables
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # Assumes tests/ is directly under project_root/

    INPUT_PATH="$PROJECT_ROOT/assets/test_images/Elon_Musk.jpg"
    MODEL_PATH="$PROJECT_ROOT/ckpts/slim_face_efficientnet_v2_s_full_model.pth"
    CONFIG_PATH="$PROJECT_ROOT/configs/image_classification_models_config.yaml"
    SRC_DIR="$PROJECT_ROOT/src/slim_face/inference"

    # Check if required files exist
    check_file "$INPUT_PATH"
    check_file "$MODEL_PATH"
    check_file "$CONFIG_PATH"
    check_directory "$SRC_DIR"

    # Check requirements
    check_requirements

    # Run inference
    log_message "Running inference..."
    python3 "${SRC_DIR}/inference.py" \
        --input_path "$INPUT_PATH" \
        --model_path "$MODEL_PATH" \
        -- "$CONFIG_PATH" || {
        log_message "ERROR: Inference failed"
        exit 1
    }

    log_message "Inference pipeline completed successfully"
}

# Trap Ctrl+C and exit gracefully
trap 'log_message "Script interrupted by user"; exit 1' INT

# Execute main function
main "$@"