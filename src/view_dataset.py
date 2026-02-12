import pandas as pd
from sklearn.datasets import load_breast_cancer

def main():
    data = load_breast_cancer()
    print(data.target_names)  # ['malignant' 'benign']

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["target_name"] = [data.target_names[i] for i in data.target]

    print("Shape:", df.shape)
    print("Targets:", list(data.target_names))
    print(df["target_name"].value_counts(), "\n")
    print(df.head(10))

if __name__ == "__main__":
    main()
