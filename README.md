# Cancer Detection with Logistic Regression (From Scratch)

Binary classification on the sklearn Breast Cancer dataset using a logistic regression model trained **from scratch** (NumPy). The project is structured like a small real-world ML repo: separate training, evaluation, artifact saving/loading, and basic plots.

## What I implemented from scratch
Core model + training math (no sklearn model used):
- Linear score: `z = Xw + b`
- Sigmoid probability: `p = 1 / (1 + exp(-z))`
- Log loss (cross-entropy) with optional L2 regularization
- Gradients for `w` and `b`
- Gradient descent training loop

## What was built with guidance
Repo structure and “project plumbing” were created with assistance (e.g., saving/loading artifacts, evaluation script layout, metrics/plots wiring, and general organization).

## Dataset
Uses `sklearn.datasets.load_breast_cancer`:
- 569 samples, 30 numeric features
- Targets:
  - `0 = malignant`
  - `1 = benign`

No external downloads required.

## Project structure
```text
cancer-detection-logreg-scratch/
├─ README.md
├─ requirements.txt
├─ reports/
│  └─ report.md
├─ src/
│  ├─ __init__.py
│  ├─ preprocess.py
│  ├─ logreg_scratch.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ view_dataset.py
├─ artifacts/   (generated, gitignored)
└─ plots/       (generated, gitignored)
```


## Setup
Create and activate a virtual environment, then install dependencies:
```bash
pip install -r requirements.txt
```

## Run 
Preview dataset:

```bash
python -m src.view_dataset
```

Train (saves artifacts/model.npz):

```bash
python -m src.train
```

Evaluate (loads artifact, prints metrics, saves plots):
```bash

python -m src.evaluate
```

## Outputs

Generated locally (not committed):

- artifacts/model.npz

    Contains w, b, standardization stats (mu, sd), loss_history, and split/training settings.

- plots/training_loss.png

    Training loss curve.

### Notes on evaluation and safety tradeoffs

This is a medical screening-style problem where missing malignant cases is the most dangerous error. Threshold choice controls the tradeoff between:

- false negatives (malignant predicted benign)

- false positives (benign predicted malignant)

The default threshold is 0.5, but it can be adjusted in evaluate.py.