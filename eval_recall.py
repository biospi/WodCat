from pathlib import Path
import typer
import json
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def main(
    input_folder: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    )
):
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


if __name__ == "__main__":
    #"E:\Cats\paper_13\All_100_10_030_008\rbf\_LeaveOneOut\fold_data"
    typer.run(main)
