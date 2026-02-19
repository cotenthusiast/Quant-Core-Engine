Logistic Regression From Scratch (NumPy)
========================================

A clean, from-scratch implementation of logistic regression using vectorized NumPy. The project demonstrates the full pipeline for binary classification: preprocessing, training with mini-batch gradient descent, evaluation, and reproducible artifact generation. The implementation is validated on a 30-feature diagnostic dataset to illustrate end-to-end performance and convergence behavior.

Note: The dataset is used only as a benchmark to validate the implementation. This repository is a learning and engineering project and is not intended for clinical use.

Core Features
-------------

Implemented from first principles to demonstrate understanding of both the mathematics and engineering of probabilistic classification:

-   Vectorized Computation\
    Model operations are expressed in matrix form (z = Xw + b) to avoid Python loops and leverage optimized linear algebra routines.

-   Mini-Batch Gradient Descent\
    Stochastic optimization with per-epoch shuffling to balance convergence stability and computational efficiency.

-   Binary Cross-Entropy Loss\
    Negative log-likelihood objective with numerically stable clipping to prevent log(0).

-   L2 Regularization (Ridge)\
    Penalizes large weights to improve generalization and reduce overfitting in higher-dimensional feature spaces.

-   Standardization Pipeline\
    Train-set z-score normalization using learned mean and standard deviation, persisted and reused at inference time to prevent data leakage.

-   Probabilistic Output\
    Produces calibrated probabilities p in [0, 1], enabling threshold tuning and sensitivity/precision trade-off analysis.

-   Reproducibility\
    Deterministic data splits, fixed seeds, and saved preprocessing statistics ensure results can be reproduced exactly.

Project Structure
-----------------

The repository follows a modular structure to separate concerns and support reproducibility:

```
logistic-regression-from-scratch/
├─ src/               # Core model, preprocessing, utilities
├─ scripts/           # Training and evaluation entry points
├─ artifacts/         # Saved model parameters and metadata
├─ plots/             # Generated figures (loss curves, ROC, confusion matrix)
├─ reports/           # Analysis notes and experiment summaries (optional)
└─ requirements.txt   # Dependencies

```

Model Overview
--------------

Logistic regression models the probability of the positive class as:

p = sigmoid(Xw + b)

where:

-   w = learned weights

-   b = bias term

-   sigmoid(z) = 1 / (1 + exp(-z))

The objective minimized during training is the regularized binary cross-entropy:

J(w, b) = -(1/m) * sum[ y log(p) + (1 - y) log(1 - p) ] + (lambda/2) * ||w||^2

where m is the batch size and lambda controls the L2 penalty strength.

Performance
-----------

The model converges rapidly on the benchmark dataset, reaching a final log-loss of approximately 0.097 within 100 epochs using default hyperparameters. Exact metrics and plots can be reproduced by running the evaluation script.

Runtime is hardware-dependent, but vectorized NumPy operations keep training efficient for moderate-sized tabular datasets.

Dataset Label Note
------------------

For the benchmark dataset:

-   y = 0 represents malignant

-   y = 1 represents benign

Evaluation metrics treat class "1" as the positive class unless labels are flipped during analysis.

Threshold Analysis
------------------

Different applications require different error trade-offs. The evaluation pipeline supports threshold sweeps so sensitivity, specificity, precision, and recall can be analyzed across operating points.

This is useful for understanding:

-   false positive vs false negative trade-offs

-   model calibration behavior

-   decision boundary sensitivity

Running the Project
-------------------

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:
```bash
python -m scripts.train
```
Evaluate and generate plots:
```bash
python -m scripts.evaluate
```
Artifacts
---------

The training process produces a compressed NumPy artifact containing:

-   learned weights and bias

-   preprocessing statistics (mean and standard deviation)

-   hyperparameters

-   loss history

-   runtime metadata

These artifacts enable deterministic reproduction of results and consistent inference on new data.

Why This Project Matters
------------------------

This repository demonstrates:

-   understanding of probabilistic modeling fundamentals

-   implementation of gradient-based optimization from scratch

-   awareness of data leakage and preprocessing correctness

-   reproducible experiment structure

-   clean software organization for ML workflows

Disclaimer
----------

This project is for educational and demonstration purposes only. It is not intended for medical or production deployment.
