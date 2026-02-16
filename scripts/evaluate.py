# scripts/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.datasets import load_breast_cancer

from src.preprocess import load_data
from src.logreg_scratch import LogisticRegression


def evaluate_and_plot(
    model_path="artifacts/model.npz",
    test_size=0.25,
    random_state=42,
    out_dir="plots",
    threshold=0.5,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load Engine + Data
    model = LogisticRegression.load(model_path)
    _, X_test, _, y_test = load_data(test_size=test_size, random_state=random_state)

    # 2) Performance Metrics (IMPORTANT: use public API on RAW X_test)
    p_test = model.predict_proba(X_test)
    preds = (p_test >= float(threshold)).astype(np.int64)

    acc = float(np.mean(preds == y_test))
    auc = float(roc_auc_score(y_test, p_test))
    loss_hist = model.loss_history

    print("=" * 45)
    print(" QUANT-CORE ENGINE: SYSTEM EVALUATION ")
    print("=" * 45)

    if model.train_time is not None:
        print(f"Training Latency:    {model.train_time:.6f} seconds")

    print(f"Test Accuracy:       {acc:.4f}")
    print(f"ROC-AUC Score:       {auc:.4f}")

    if loss_hist is not None and len(loss_hist) > 0:
        print(f"Final Training Loss: {loss_hist[-1]:.6f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # 3) Feature Importance
    data = load_breast_cancer()
    names = data.feature_names
    weights = model.w
    idx = np.argsort(np.abs(weights))[::-1][:10]

    print("\nTop 10 Feature Weights (Alpha Factors):")
    print("-" * 45)
    for i in idx:
        print(f"{names[i]:25s} | Weight: {weights[i]: .4f}")

    # 4) Visualization Suite

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
    plt.title("Confusion Matrix (Type I/II Error Analysis)")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Loss Curve
    if loss_hist is not None and len(loss_hist) > 0:
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(loss_hist)), loss_hist)
        plt.title("Model Convergence (Mini-batch Gradient Descent)")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss (L2 Regularized)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, p_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[Success] Visualizations archived in: {out_dir}/")


if __name__ == "__main__":
    evaluate_and_plot()