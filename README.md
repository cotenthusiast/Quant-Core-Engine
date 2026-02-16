Quant-Core-Engine: Stochastic Logistic Regression
=================================================

A performance-optimized implementation of a Logistic Regression engine built from scratch using vectorized NumPy. While demonstrated on high-dimensional clinical diagnostic data, the core architecture is designed as a reusable, quant-style classification component for risk-sensitive binary prediction, probability scoring, and feature-weight interpretation.

Note: This repository uses a medical dataset as a benchmark for validating the engine end-to-end. It is a learning and engineering demonstration project and is not intended for clinical use.

Core Quantitative Features
--------------------------

Implemented from the ground up to demonstrate mastery of the mathematics and engineering of predictive modeling:

-   Vectorized Execution: Core operations are expressed in matrix form (z = Xw + b) to reduce Python overhead and leverage optimized linear algebra routines.

-   Stochastic Optimization: Mini-Batch Gradient Descent with per-epoch shuffling to balance convergence stability and computational efficiency.

-   L2 Regularization (Ridge): Adds a penalty term to control weight growth and mitigate overfitting in high-dimensional settings (30 features).

-   Standardization Pipeline: Train-set z-score normalization using learned mu/sd, persisted and reused at inference time to prevent leakage and maintain numerical stability.

-   Probabilistic Output: Produces continuous probability scores p in [0, 1], enabling dynamic thresholding, sensitivity tuning, and cost-of-error analysis.

-   High-Precision Latency Benchmarking: Records training duration using time.perf_counter() for high-resolution performance benchmarking (resolution depends on system and OS).

Project Architecture
--------------------

The repository follows modular SWE structure for reproducibility and clean separation of concerns:
```
quant-core-engine/
├─ src/               # Core engine logic (standardization, model math, save/load)\
├─ scripts/           # Pipeline execution (train, evaluate)\
├─ artifacts/         # Model persistence (serialized weights, scaling stats, metadata)\
├─ plots/             # Performance visualization (loss, ROC, confusion matrix)\
├─ reports/           # Quantitative analysis and threshold notes (optional)\
└─ requirements.txt   # Dependency management
```
Quantitative Performance
------------------------

The engine demonstrates rapid convergence on the benchmark dataset, reaching a finalized log-loss of approximately 0.097 within 100 epochs (with default hyperparameters). Run the evaluation script to reproduce exact metrics and plots on your machine.

Important label note (dataset):

-   y = 0 represents malignant

-   y = 1 represents benign\
    ROC/AUC and thresholding are therefore "positive = benign" unless you explicitly flip the labeling for a malignancy-detection view.

Risk Calibration & Threshold Sweeps
-----------------------------------

In many risk-sensitive domains, the cost of a False Negative (Type II error) can outweigh a False Positive. This engine includes an evaluation suite that supports threshold sweeps so sensitivity can be tuned to the domain's cost-of-error requirements.

Running the Engine
------------------

1.  Initialize Environment:

pip install -r requirements.txt

1.  Execute Training Pipeline:

python -m scripts.train

1.  Run Sensitivity Analysis & Evaluation:

python -m scripts.evaluate

Why this matters for Quant Roles
--------------------------------

-   Feature Importance: The learned weights w can be inspected directly to rank the strongest predictive factors, supporting interpretability and feature-driven investigation.

-   Numerical Stability: Custom log-loss implementation with clipping safeguards to prevent overflow/underflow and improve robustness on extreme scores.

-   Reproducibility and Leakage Safety: Deterministic data splits and train-only scaling statistics ensure consistent, audit-friendly results.

-   Scalability: The modular src/ architecture makes the engine easy to swap into larger pipelines (batch scoring, walk-forward validation, threshold calibration) with minimal friction.

Disclaimer
----------

This is a learning and demonstration project. It is not intended for clinical deployment or medical decision-making.