```markdown
# Running Inference Test Script

Instructions to run the `tests/inference_test.sh` script in the `tests` folder on Linux, Windows, and macOS.

## Prerequisites
1. Install Python, PyTorch, Accelerate:
   ```bash
   pip install -r requirements/requirements.txt
   ```
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```
3. Make script executable (Linux/macOS):
   ```bash
   chmod +x tests/inference_test.sh
   ```

## Linux
1. Open terminal, go to folder:
   ```bash
   cd tests
   ```
2. Run script:
   ```bash
   ./inference_test.sh
   ```
3. **Fix issues**:
   - Use `bash inference_test.sh` if `./` fails.
   - Fix line endings:
     ```bash
     sudo apt install dos2unix
     dos2unix inference_test.sh
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
   chmod +x inference_test.sh
   ```
5. Run script:
   ```bash
   ./inference_test.sh
   ```
6. **Fix issues**:
   - Fix line endings:
     ```bash
     sudo apt install dos2unix
     dos2unix inference_test.sh
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
   chmod +x inference_test.sh
   ```
4. Run script:
   ```bash
   ./inference_test.sh
   ```
5. **Fix issues**:
   - Fix line endings:
     ```bash
     brew install dos2unix
     dos2unix inference_test.sh
     ```

## Notes
- Ensure GPU support (CUDA for Linux/Windows, MPS for macOS) if needed.
- Check script for extra settings (e.g., `export CUDA_VISIBLE_DEVICES=0`).
- Save output:
  ```bash
  ./inference_test.sh > output.log 2>&1
  ```
```