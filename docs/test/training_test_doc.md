# Running Training Scripts

Instructions to run these scripts in the `tests` folder on Linux, Windows, and macOS:
- `tests/training_accelerate_efficientnet_b3.sh`
- `tests/training_accelerate_efficientnet_v2_s.sh`
- `tests/training_accelerate_regnet_y_800mf.sh`
- `tests/training_accelerate_vit_b_16_test.sh`

## Prerequisites
1. Install Python, PyTorch, Accelerate:
   ```bash
   pip install requirements/requirements.txt
   ```
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```
3. Make scripts executable (Linux/macOS):
   ```bash
   chmod +x tests/*.sh
   ```

## Linux
1. Open terminal, go to folder:
   ```bash
   cd tests
   ```
2. Run scripts:
   ```bash
   ./training_accelerate_efficientnet_b3.sh
   ./training_accelerate_efficientnet_v2_s.sh
   ./training_accelerate_regnet_y_800mf.sh
   ./training_accelerate_vit_b_16_test.sh
   ```
3. **Fix issues**:
   - Use `bash training_accelerate_efficientnet_b3.sh` if `./` fails.
   - Fix line endings:
     ```bash
     sudo apt install dos2unix
     dos2unix training_accelerate_*.sh
     ```

## Windows (using WSL)
1. Install WSL and Ubuntu from Microsoft Store.
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   pip install -r requirements/requirements.txt
   ```
3. Go to folder:
   ```bash
   cd ./tests
   ```
4. Make executable:
   ```bash
   chmod +x training_accelerate_*.sh
   ```
5. Run scripts:
   ```bash
   ./training_accelerate_efficientnet_b3.sh
   ```
6. **Fix issues**:
   - Fix line endings:
     ```bash
     sudo apt install dos2unix
     dos2unix training_accelerate_*.sh
     ```

## macOS
1. Open Terminal, go to folder:
   ```bash
   cd tests
   ```
2. Install dependencies:
   ```bash
   brew install python
   pip install -r requirements/requirements.txt
   ```
3. Make executable:
   ```bash
   chmod +x training_accelerate_*.sh
   ```
4. Run scripts:
   ```bash
   ./training_accelerate_efficientnet_b3.sh
   ```
5. **Fix issues**:
   - Fix line endings:
     ```bash
     brew install dos2unix
     dos2unix training_accelerate_*.sh
     ```

## Notes
- Ensure GPU support (CUDA for Linux/Windows, MPS for macOS) if needed.
- Check scripts for extra settings (e.g., `export CUDA_VISIBLE_DEVICES=0`).
- Save output:
  ```bash
  ./training_accelerate_efficientnet_b3.sh > output.log 2>&1
  ```
