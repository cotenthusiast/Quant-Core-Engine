import os
import numpy as np

from src.preprocess import load_data, standardize_fit, standardize_apply
from src.logreg_scratch import log_loss, predict_proba, gradients

def train_model(epochs=1000, lr=0.1, l2=0.01, log_every=200,
                test_size=0.25, random_state=42,
                model_path="artifacts/model.npz"):
    X_train, X_test, y_train, y_test = load_data(test_size=test_size, random_state=random_state)

    mu, sd = standardize_fit(X_train)
    X_train_s = standardize_apply(X_train, mu, sd)

    w = np.zeros(X_train_s.shape[1], dtype=np.float64)
    b = 0.0
    loss_history = np.empty(epochs, dtype=np.float64)

    for epoch in range(epochs):
        p = predict_proba(X_train_s, w, b)
        loss = log_loss(y_train, p, l2=l2, w=w)
        loss_history[epoch] = loss

        grad_w, grad_b = gradients(X_train_s, y_train, p, l2=l2, w=w)
        w -= lr * grad_w
        b -= lr * grad_b

        if log_every and epoch % log_every == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    np.savez(
        model_path,
        w=w,
        b=np.array(b),
        mu=mu,
        sd=sd,
        loss_history=loss_history,
        test_size=np.array(test_size),
        random_state=np.array(random_state),
        epochs=np.array(epochs),
        lr=np.array(lr),
        l2=np.array(l2),
    )

    return model_path

if __name__ == "__main__":
    train_model()
