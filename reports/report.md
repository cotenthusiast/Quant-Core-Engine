Binary Classification: Logistic Regression From Scratch (NumPy)
===============================================================

1\. Executive Summary
---------------------

This project implements a binary classifier from scratch using vectorized NumPy operations. The model is a logistic regression trained with mini-batch stochastic gradient descent (SGD), evaluated on a 30-feature diagnostic dataset. On the held-out test set, it achieved **98.60% accuracy** and **0.9971 ROC-AUC**. The repository emphasizes clean preprocessing, numerical stability, reproducibility, and artifact generation (metrics, plots, saved parameters).

2\. Model and Optimization
--------------------------

The classifier uses logistic regression with mini-batch SGD.

-   Linear score: z = Xw + b

-   Sigmoid probability: p = sigmoid(z) = 1 / (1 + exp(-z))

-   Loss: binary cross-entropy (log loss)

-   Regularization: L2 (ridge) penalty on weights to reduce overfitting and stabilize coefficients

Objective:

J(w, b) = -(1/n) * sum( y * log(p) + (1 - y) * log(1 - p) ) + (lambda/2) * ||w||^2

3\. Data Pipeline and Numerical Stability
-----------------------------------------

To ensure stable gradients and fair evaluation, the pipeline follows standard best practices:

-   Standardization (z-score):

    -   Fit mean (mu) and standard deviation (sigma) on the training split only

    -   Apply the same (mu, sigma) to validation/test splits to avoid leakage

-   Probability safety:

    -   Clip probabilities to [epsilon, 1 - epsilon] before log loss to prevent log(0)

These measures improve convergence behavior and make results reproducible and comparable across runs.

4\. Training Dynamics
---------------------

Training used:

-   Learning rate (eta): 0.1

-   Batch size: 32

-   Epochs: 100 (reported convergence by ~90)

Observed behavior:

-   Initial average loss: 0.3449 (epoch 0)

-   Final average loss: 0.0975 (epoch ~90)

Loss decreased smoothly under mini-batch noise, indicating stable optimization.

5\. Performance and Runtime Notes
---------------------------------

The implementation is fully vectorized and avoids Python loops in the critical path.

-   Timing method: time.perf_counter() (monotonic, high resolution)

-   End-to-end training time (100 epochs, 30 features): ~0.0268 seconds on the test machine

This runtime is hardware-dependent; the key point is that vectorization keeps training fast and scalable for moderate-sized tabular datasets.

6\. Thresholding and Error Trade-offs
-------------------------------------

Accuracy and ROC-AUC measure ranking and overall discrimination, but real deployments often care about asymmetric error costs.

This project supports threshold analysis:

-   Default threshold: 0.5

-   Alternative thresholds can be chosen to emphasize recall/sensitivity or precision, depending on the cost of false negatives vs false positives.

Recommended reporting practice:

-   Include confusion matrices at multiple thresholds (e.g., 0.3, 0.5, 0.7)

-   Track precision, recall, F1, and ROC-AUC together

-   Discuss where the model fails (error analysis), not just aggregate metrics

Note: If you include a confusion matrix in this report, it must match the stated accuracy. A confusion matrix where every prediction is the majority class would not be consistent with 98.60% accuracy. Regenerate and paste the exact matrix produced by your evaluation script.

7\. Persistence and Reproducibility
-----------------------------------

The trained model is serialized to a compressed NumPy artifact (artifacts/model.npz) containing:

-   Parameters: final weights (w) and bias (b)

-   Preprocessing metadata: training mean (mu) and std (sigma)

-   Training metadata: learning rate, L2 strength, batch size, epoch count

-   Diagnostics: loss history for plotting and auditability

-   Runtime note: recorded training duration for reference (hardware-dependent)

Reproducibility controls:

-   Fixed random seed(s)

-   Deterministic data split strategy

-   Saved preprocessing statistics to ensure consistent out-of-sample behavior
