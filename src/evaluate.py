import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score, roc_curve, roc_curve
from sklearn.datasets import load_breast_cancer

from src.preprocess import load_data, standardize_apply
from src.logreg_scratch import predict_proba

def load_results(model_path="artifacts/model.npz"):
    return np.load(model_path)

def evaluate(model, threshold=0.5, plot_path="plots/training_loss.png"):
    w = model["w"]
    b = float(model["b"])          # scalar safety
    mu = model["mu"]
    sd = model["sd"]
    loss_history = model["loss_history"]

    # recreate the same split deterministically
    test_size = float(model["test_size"])
    random_state = int(model["random_state"])
    _, X_test, _, y_test = load_data(test_size=test_size, random_state=random_state)

    X_test_s = standardize_apply(X_test, mu, sd)
    p_test = predict_proba(X_test_s, w, b)
    predictions = (p_test >= threshold).astype(int)

    print("Test Accuracy:", np.mean(predictions == y_test))
    print("ROC-AUC:", roc_auc_score(y_test, p_test))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, digits=3))

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # feature importance
    data = load_breast_cancer()
    names = data.feature_names
    idx = np.argsort(np.abs(w))[::-1][:10]  # top 10
    print("\nTop 10 features by |weight| (standardized):")
    for i in idx:
        print(f"{names[i]:25s}  w={w[i]: .4f}")

    # threshold sweep
    print("\nThreshold sweep (focus: reduce malignant→benign misses):")
    for t in [0.3, 0.5, 0.7, 0.9]:
        preds_t = (p_test >= t).astype(int)
        cm = confusion_matrix(y_test, preds_t)
        misses_malignant = cm[0, 1]   # malignant predicted benign
        false_alarms = cm[1, 0]       # benign predicted malignant
        acc = (preds_t == y_test).mean()
        print(f"t={t:.1f}  acc={acc:.3f}  misses_malignant={misses_malignant}  false_alarms={false_alarms}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    plt.figure()
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    model = load_results()
    evaluate(model)

if __name__ == "__main__":
    main()
