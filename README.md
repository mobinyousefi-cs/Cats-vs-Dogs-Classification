# Cats vs Dogs Classification using CNN (Keras)

A clean, modular deep learning project for binary image classification on the classic **Dogs vs Cats (Asirra)** dataset using **TensorFlow/Keras**.

Author: **Mobin Yousefi**  \
GitHub: <https://github.com/mobinyousefi-cs>

---

## 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images of **cats** and **dogs**. It is designed as a beginner-friendly yet production-style codebase:

- Clear **src/** layout with a Python package (`cats_vs_dogs`)
- Dedicated modules for **configuration**, **data preparation**, **model definition**, **training**, **evaluation**, and **inference**
- Reproducible experiments using a single configuration dataclass
- Ready for GitHub with **MIT License**, **CI (GitHub Actions)**, and **tests**

The code is written in a clean, documented, and extensible way, suitable for:

- Deep learning beginners who want a solid template
- Students building a GitHub portfolio
- Fast experimentation on binary image classification tasks

---

## 2. Dataset

We use the classic **Dogs vs Cats (Asirra)** dataset from Kaggle:

> https://www.kaggle.com/c/dogs-vs-cats/data

### 2.1. Raw Dataset Structure (Kaggle)

After downloading and unzipping `train.zip` from Kaggle, you will have images in a single folder:

```text
train/
    cat.0.jpg
    cat.1.jpg
    ...
    dog.0.jpg
    dog.1.jpg
    ...
```

Each filename starts with `cat` or `dog`, which is used as the label.

### 2.2. Project Data Layout

Inside this repository, we use the following layout:

```text
cats-vs-dogs-classifier/
├── data/
│   ├── raw/
│   │   └── train/           # All original images from Kaggle train.zip
│   └── processed/
│       ├── train/
│       │   ├── cats/
│       │   └── dogs/
│       ├── val/
│       │   ├── cats/
│       │   └── dogs/
│       └── test/
│           ├── cats/
│           └── dogs/
```

The script `cats_vs_dogs.data` will:

- Read images from `data/raw/train/`
- Split them into **train / validation / test** subsets
- Organize them into class-based folders under `data/processed/`

Default splits:

- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

You can adjust these via the configuration if needed.

---

## 3. Project Structure

```text
cats-vs-dogs-classifier/
├── src/
│   └── cats_vs_dogs/
│       ├── __init__.py
│       ├── config.py        # Central configuration (paths, hyperparameters)
│       ├── data.py          # Data preparation & tf.data pipelines
│       ├── model.py         # CNN model definition (Keras)
│       ├── train.py         # Training loop & callbacks
│       ├── evaluate.py      # Evaluation on test dataset
│       └── predict.py       # Inference on single images / folders
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py       # Basic configuration tests
│   └── test_model.py        # Sanity checks for model construction
│
├── notebooks/
│   └── 00_quickstart.ipynb  # (Optional) Exploration / experimentation notebook
│
├── data/                    # (Created by you; ignored by Git)
├── models/                  # Saved models (.keras, .h5, etc.)
├── logs/                    # TensorBoard logs
│
├── pyproject.toml           # Project metadata & dependencies
├── requirements.txt         # Convenience requirements file
├── .gitignore
├── .editorconfig
├── LICENSE                  # MIT License
└── .github/
    └── workflows/
        └── ci.yml           # CI: Ruff, Black, Pytest
```

---

## 4. Installation & Setup

### 4.1. Clone the repository

```bash
git clone https://github.com/mobinyousefi-cs/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```

### 4.2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# or
.venv\Scripts\activate        # Windows
```

### 4.3. Install dependencies

You can install via `pyproject.toml` (recommended):

```bash
pip install -e .[dev]
```

Or using `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow can be heavy. On some systems you may prefer `tensorflow-cpu` instead of `tensorflow`.

---

## 5. Preparing the Dataset

1. Download `train.zip` from the Kaggle Dogs vs Cats competition.
2. Create the folder structure:

   ```bash
   mkdir -p data/raw/train
   ```

3. Extract all images into `data/raw/train/` so that you have files like `data/raw/train/cat.0.jpg`, `data/raw/train/dog.0.jpg`, etc.

4. Run the data preparation script to create train/val/test splits:

   ```bash
   python -m cats_vs_dogs.data \
       --raw-dir data/raw/train \
       --output-dir data/processed \
       --val-split 0.15 \
       --test-split 0.15
   ```

This will create the `data/processed/` structure with `train`, `val`, and `test` folders, each containing `cats/` and `dogs/` subfolders.

---

## 6. Training the Model

Once the dataset is prepared, you can start training.

### 6.1. Basic training command

```bash
python -m cats_vs_dogs.train --epochs 15
```

What this does:

- Loads the processed datasets from `data/processed/`
- Builds a simple but effective CNN using Keras
- Trains the model with **early stopping** and **model checkpointing**
- Saves the best model to `models/cats_vs_dogs_cnn.keras`
- Logs metrics and loss for TensorBoard under `logs/`

### 6.2. Run TensorBoard (optional)

```bash
tensorboard --logdir logs
```

Open the provided URL in your browser to inspect loss and accuracy curves.

---

## 7. Evaluating the Model

To evaluate the model on the **test** dataset:

```bash
python -m cats_vs_dogs.evaluate \
    --model-path models/cats_vs_dogs_cnn.keras
```

This will:

- Load the trained model
- Build the test dataset from `data/processed/test`
- Print metrics such as **loss** and **accuracy**

You can extend `evaluate.py` to add confusion matrices, ROC curves, or detailed classification reports.

---

## 8. Making Predictions

Use the `predict.py` module to run inference on a single image or on all images in a folder.

### 8.1. Predict a single image

```bash
python -m cats_vs_dogs.predict \
    --model-path models/cats_vs_dogs_cnn.keras \
    --image path/to/your_image.jpg
```

Example output:

```text
Image: path/to/your_image.jpg
Predicted class: dog
Confidence: 0.93
```

### 8.2. Predict all images in a folder

```bash
python -m cats_vs_dogs.predict \
    --model-path models/cats_vs_dogs_cnn.keras \
    --folder path/to/folder
```

You can redirect the output to a file if needed:

```bash
python -m cats_vs_dogs.predict \
    --model-path models/cats_vs_dogs_cnn.keras \
    --folder samples/ > predictions.txt
```

---

## 9. Running Tests

The project includes a few basic tests to ensure that configurations and the model can be instantiated correctly.

Run tests with:

```bash
pytest
```

This will:

- Verify that the `TrainingConfig` initializes without errors
- Check that the Keras model can be built and has the expected output shape

You can add more tests (e.g., for data loading) as the project evolves.

---

## 10. Continuous Integration (CI)

A GitHub Actions workflow is provided under `.github/workflows/ci.yml`.

On every push / pull request, it will:

1. Install the project (including dev dependencies)
2. Run **Ruff** (linter)
3. Run **Black** in check mode (code formatter)
4. Run **Pytest**

This helps keep the codebase clean and stable as you extend it.

---

## 11. Configuration & Hyperparameters

All core settings are defined in `cats_vs_dogs.config.TrainingConfig`, including:

- Image size (default: `180x180`)
- Batch size (default: `32`)
- Train/val/test splits
- Learning rate
- Number of epochs
- Important directory paths (raw data, processed data, models, logs)

You can:

- Edit defaults directly in `config.py`, or
- Override some of them via CLI arguments in `train.py`, `data.py`, etc.

---

## 12. Extending the Project

Here are some ideas to extend this project:

- Replace the simple CNN with **transfer learning** (e.g., `MobileNetV2`, `EfficientNet`)
- Add **data augmentation** layers (RandomFlip, RandomRotation, etc.)
- Use **learning rate schedules** or advanced callbacks
- Implement **k-fold cross-validation**
- Export the model to **TensorFlow Lite** or **ONNX** for deployment

The current structure is intentionally modular to make these extensions straightforward.

---

## 13. License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

You are free to use, modify, and distribute this code with proper attribution.

