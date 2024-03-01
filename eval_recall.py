from pathlib import Path
import typer
import json
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd


def eval_recall(
    input_folder: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    )
):
    print("Find optimal Sensitivity/Specificity...")
    print(input_folder)
    fold_data_files = list(input_folder.glob("*.json"))
    y_true_list = []
    y_score_list = []

    for file in fold_data_files:
        print(file)
        with open(file, "r") as fp:
            fold_data = json.load(fp)
            y_true = fold_data["y_test"]
            y_true_list.extend(y_true)
            y_score = fold_data["y_pred_proba_test"]
            y_score_list.extend(y_score)

    fpr, tpr, thresholds = roc_curve(y_true_list, y_score_list)
    sensitivity = tpr
    specificity = 1 - fpr

    # Find the index of the point on the ROC curve closest to the top-left corner
    optimal_idx = np.argmax(sensitivity + specificity)

    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = sensitivity[optimal_idx]
    optimal_specificity = specificity[optimal_idx]

    print("Optimal Threshold:", optimal_threshold)
    print("Optimal Sensitivity:", optimal_sensitivity)
    print("Optimal Specificity:", optimal_specificity)

    auc = roc_auc_score(y_true_list, y_score_list)
    print("AUC:", auc)
    df = pd.DataFrame()
    df["auc"] = [auc]
    df["optimal_threshold"] = [optimal_threshold]
    df["optimal_sensitivity"] = [optimal_sensitivity]
    df["optimal_specificity"] = [optimal_specificity]
    filepath = input_folder.parent / "recall_data.csv"
    print(filepath)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    #"E:\Cats\paper_13\All_100_10_030_008\rbf\_LeaveOneOut\fold_data"
    typer.run(eval_recall)
