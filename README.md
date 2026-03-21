## Medical Image Diagnosis System

This project focuses **only on preprocessing** for three MedMNIST benchmark datasets:

- **ChestMNIST** – chest X‑ray classification
- **RetinaMNIST** – fundus image classification
- **DermaMNIST** – skin lesion classification

Each dataset has its **own preprocessing pipeline**, organized in a structured folder layout to keep experiments clean and modular. No model training code is included here.

### Project Structure

- **`data/`** – location where MedMNIST data will be downloaded / cached (handled automatically by `medmnist`).
- **`src/`**
  - **`datasets/`**
    - MedMNIST dataset wrappers and preprocessing transforms.
  - **`preprocessing/`**
    - Scripts that download and preprocess ChestMNIST, RetinaMNIST, and DermaMNIST (e.g., saving processed arrays to disk).
- **`requirements.txt`** – Python dependencies.

### Datasets

The project uses the `medmnist` Python package to automatically download and manage:

- `ChestMNIST` – `ChestMNIST` class
- `RetinaMNIST` – `RetinaMNIST` class
- `DermaMNIST` – `DermaMNIST` class

### How to Get Started

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run any of the preprocessing scripts, for example:

```bash
python -m src.preprocessing.preprocess_chestmnist
```

