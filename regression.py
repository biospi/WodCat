import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file = Path(os.getcwd()) / "metadata.csv"
    df = pd.read_csv(file, sep=",", nrows=55)
    print(df)
    X = np.array([[x] for x in df["Age"].values])
    y = df["Status"].values

    log_reg_model = LogisticRegression()

    n_splits = 5
    n_repeats = 10
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        print(f"{i}/{n_splits*n_repeats}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        log_reg_model.fit(X_train, y_train)

        probas_ = log_reg_model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3)

    # Plot median ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(
        mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {mean_auc:.2f})", lw=2
    )
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    filepath = 'roc_curve.png'
    print(filepath)
    plt.savefig(filepath)
    plt.close()
