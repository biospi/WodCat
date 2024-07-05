import random
from pathlib import Path
import typer
import json
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
from tqdm import tqdm



def test():
    input_dir = Path("/mnt/storage/scratch/axel/cats/paper_debug_regularisation_36")
    folders = list(input_dir.glob("**/fold_data"))

    recall_data = list(input_dir.glob("**/recall_data.csv"))
    dfs = []
    for file in recall_data:
        df = pd.read_csv(file)
        df["dataset"] = file
        dfs.append(df)

    data = pd.concat(dfs)
    data = data.sort_values(["optimal_sensitivity", "optimal_specificity"])

    dir_list, auc_list, optimal_threshold_list, optimal_sensitivity_list, optimal_specificity_list = [], [], [], [], []
    for folder in folders:
        auc, optimal_threshold, optimal_sensitivity, optimal_specificity = eval_recall(folder)
        auc_list.append(auc)
        optimal_threshold_list.append(optimal_threshold)
        optimal_sensitivity_list.append(optimal_sensitivity)
        optimal_specificity_list.append(optimal_specificity)
        dir_list.append(folder)

    df = pd.DataFrame()
    df["auc"] = auc_list
    df["optimal_threshold"] = optimal_threshold_list
    df["optimal_sensitivity"] = optimal_sensitivity_list
    df["optimal_specificity"] = optimal_specificity_list
    df["directory"] = dir_list

    df = df.sort_values(["auc", "optimal_sensitivity", "optimal_specificity"])
    print(df)
    filepath = input_dir / "recall_test.csv"
    print(filepath)
    df.to_csv(filepath, index=False)


def get_cli(data, l="Sensitivity"):
    print(f"DATA={data}")
    # Convert the list to a NumPy array for convenience
    data = np.array(data)
    # Calculate the median
    median = np.nanmedian(data)
    # Calculate the 2.5th and 97.5th percentiles for the confidence interval
    lower_bound = np.nanpercentile(data, 2.5)
    upper_bound = np.nanpercentile(data, 97.5)
    print(f"Median {l}: {median:.4f}")
    print(f"95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")


def eval_recall(
    input_folder: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    n_boots: int = 10
):
    optimal_sensitivity_list, optimal_specificity_list = [], []
    for i in range(n_boots):
        print("Find optimal Sensitivity/Specificity...")
        print(input_folder)
        fold_data_files = list(input_folder.glob("*.json"))

        num_files_to_select = int(len(fold_data_files) * 0.95)
        selected_files = random.sample(fold_data_files, num_files_to_select)

        y_true_list = []
        y_score_list = []

        # optimal_sensitivity_list = []
        # optimal_specificity_list = []

        for file in tqdm(selected_files):
            with open(file, "r") as fp:
                fold_data = json.load(fp)
                y_true = fold_data["y_test"]
                y_true_list.extend(y_true)
                y_score = fold_data["y_pred_proba_test"]
                y_score_list.extend(y_score)

        #         print(y_true)
        #         print(y_score)
        #         fpr, tpr, thresholds = roc_curve(y_true, y_score)
        #         sensitivity = tpr
        #         specificity = 1 - fpr
        #         optimal_idx = np.argmax(sensitivity + specificity)
        #         optimal_threshold = thresholds[optimal_idx]
        #         optimal_sensitivity = sensitivity[optimal_idx]
        #         optimal_specificity = specificity[optimal_idx]
        #         optimal_sensitivity_list.append(optimal_sensitivity)
        #         optimal_specificity_list.append(optimal_specificity)
        #
        # get_cli(optimal_sensitivity_list, "optimal_sensitivity")
        # get_cli(optimal_specificity_list, "optimal_specificity")

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
        optimal_sensitivity_list.append(optimal_sensitivity)
        optimal_specificity_list.append(optimal_specificity)
        #return auc, optimal_threshold, optimal_sensitivity, optimal_specificity

    get_cli(optimal_sensitivity_list, "optimal_sensitivity")
    get_cli(optimal_specificity_list, "optimal_specificity")


if __name__ == "__main__":
    #test()
    #"E:\Cats\paper_13\All_100_10_030_008\rbf\_LeaveOneOut\fold_data"
    typer.run(eval_recall)
