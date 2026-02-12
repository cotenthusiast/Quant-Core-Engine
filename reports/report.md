# Cancer Detection with Logistic Regression (From Scratch)

## Summary
Built a binary classifier for the sklearn breast cancer dataset using logistic regression implemented from scratch (NumPy). Trained with gradient descent on log loss with L2 regularization, saved artifacts, and evaluated on a held-out test split. Test accuracy was 0.9860 with one malignant→benign miss and one benign→malignant false alarm.

## Dataset
Source: `sklearn.datasets.load_breast_cancer`

- Samples: 569
- Features: 30 numeric features per sample (tumor measurements)
- Target meanings:
  - 0 = malignant
  - 1 = benign

## Train/Test Split
- Test size: 0.25
- Random seed: 42
- Stratified split: preserves class ratio in train and test

## Preprocessing
I standardized features using training-set statistics only:

For each feature column:
```text
x_scaled = (x - mean_train) / std_train
```

Reason: features have very different scales. Standardization makes gradient descent stable and makes learned weights comparable across features.

## Model
For one sample x:

- Linear score: z = w·x + b
- Probability: p = sigmoid(z) = 1 / (1 + exp(-z))

Interpretation:
p is the model’s estimated probability that the sample is class 1 (benign).

## Loss Function (Log Loss / Cross-Entropy)
For labels y ∈ {0,1} and predicted probabilities p:

Loss = -mean( y*log(p) + (1-y)*log(1-p) )

L2 regularization:
Loss += (l2/2) * sum(w^2)

Regularization discourages excessively large weights and helps generalization.

## Training Hyperparameters
- Learning rate (lr): 0.1
- Epochs: 1000
- L2 strength (l2): 0.01

## Training (Gradient Descent)
Training repeats these steps:

1) Compute probabilities p using current w and b.
2) Compute loss to measure how wrong predictions are.
3) Compute gradients (how to change w and b to reduce loss):
   e = p - y
   grad_w = (X^T @ e)/n + l2*w
   grad_b = mean(e)
4) Update parameters:
   w = w - lr*grad_w
   b = b - lr*grad_b

Sanity check:
At epoch 0 with w=0 and b=0, p=0.5 for all samples and loss ≈ 0.6931, which matched my logs.

## Evaluation Metrics
- Accuracy: fraction of correct predictions.
- Confusion matrix: shows error types.
- Precision/Recall/F1: per-class performance.
- ROC-AUC: threshold-independent measure of ranking quality.

### Error priorities (medical screening context)
Because 0 = malignant and 1 = benign:

- False negative for malignant (malignant predicted as benign) is the most dangerous mistake.
- False positive (benign predicted as malignant) is less dangerous because it triggers further testing.

## Results (Test Set)
- Accuracy: 0.9860
- ROC-AUC: 0.9971
- Confusion matrix:

```text
[[52  1]
 [ 1 89]]
```

    Matrix layout: [[true 0 predicted 0, true 0 predicted 1], [true 1 predicted 0, true 1 predicted 1]]


Interpretation:
- 1 malignant was predicted as benign (dangerous miss).
- 1 benign was predicted as malignant (false alarm).

## Training Curve
The training loss decreased from 0.6931 at epoch 0 to 0.0970773873104337 by the end, indicating stable convergence.
Saved plot: `plots/training_loss.png`.

## Threshold Tradeoff
Default threshold was 0.5 (predict benign if p >= 0.5).
Raising the threshold makes the model more conservative about predicting benign, which reduces malignant→benign mistakes at the cost of more benign→malignant false alarms.

## Artifacts
Training saves:
- w, b (model parameters)
- mu, sd (standardization stats)
- loss_history
- split settings (test_size, random_state)
Saved file: `artifacts/model.npz`

Evaluation loads the artifact and recreates the test split deterministically using the saved split settings.

## Limitations / Next Steps
- Add explicit threshold selection rule to minimize malignant false negatives.
- Compare performance against sklearn LogisticRegression as a consistency check.
