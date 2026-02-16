# scripts/train.py
from src.logreg_scratch import LogisticRegression
from src.preprocess import load_data


def train_model():
    """
    Training pipeline for scratch Logistic Regression:
    - loads data split
    - trains mini-batch GD
    - saves artifact to artifacts/model.npz
    """
    # Deterministic split for reproducibility
    X_train, X_test, y_train, y_test = load_data(test_size=0.25, random_state=42)

    model = LogisticRegression(
        lr=0.1,
        l2=0.01,
        epochs=100,
        batch_size=32,
        seed=42,
    )

    print("Initiating Mini-batch Gradient Descent...")

    model.fit(
        X_train,
        y_train,
        log_every=10,
        model_path="artifacts/model.npz",
    )

    model.summary()


if __name__ == "__main__":
    train_model()