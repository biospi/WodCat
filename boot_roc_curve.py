import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Manager
import numpy as np
import json
from tqdm import tqdm

from utils.utils import ninefive_confidence_interval, get_time_of_day
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
import sys
np.random.seed(0)
from matplotlib import rcParams
# Set matplotlib to use Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import scipy
from scipy.stats import ttest_rel
import scipy.stats as stats

class AnyObjectHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        l1 = plt.Line2D(
            [x0, y0 + width],
            [0.7 * height, 0.7 * height],
            linestyle=orig_handle[1],
            color=orig_handle[0],
        )
        l2 = plt.Line2D(
            [x0, y0 + width], [0.3 * height, 0.3 * height], color=orig_handle[0]
        )
        return [l1, l2]


def worker(
    out_dir,
    data,
    i,
    tot,
    bootstrap,
    xaxis_train,
    xaxis_test,
    auc_list_test,
    auc_list_train,
    tprs_test,
    tprs_train,
    fprs_test,
    fprs_train,
):
    all_test_y_list = []
    all_test_proba_list = []
    all_train_y_list = []
    all_train_proba_list = []
    prec_list_test = []
    prec_list_train = []
    #print(f"bootstrap results progress {i}/{tot}...")

    all_test_proba = []
    all_test_y = []
    all_train_proba = []
    all_train_y = []
    for n, filepath in enumerate(bootstrap):
        loo_result = data[filepath.stem]
        y_pred_proba_test = np.array(loo_result["y_pred_proba_test"])
        if len(y_pred_proba_test.shape) > 1:
            y_pred_proba_test = y_pred_proba_test[:, 1]
        y_pred_proba_test = y_pred_proba_test.astype(np.float16)
        y_test = loo_result["y_test"]

        all_test_proba.extend(y_pred_proba_test)
        all_test_y.extend(y_test)
        all_test_y_list.extend(y_test)
        all_test_proba_list.extend(y_pred_proba_test)

        y_pred_proba_train = np.array(loo_result["y_pred_proba_train"])
        if len(y_pred_proba_train.shape) > 1:
            y_pred_proba_train = y_pred_proba_train[:, 1]
        y_pred_proba_train = y_pred_proba_train.astype(np.float16)
        y_train = loo_result["y_train"]
        all_train_proba.extend(y_pred_proba_train)
        all_train_y.extend(y_train)
        all_train_y_list.extend(y_train)
        all_train_proba_list.extend(y_pred_proba_train)

    fpr, tpr, thresholds = roc_curve(all_test_y, all_test_proba)
    tprs_test.append(tpr)
    fprs_test.append(fpr)
    roc_auc = auc(fpr, tpr)
    #print(roc_auc, all_test_y, all_test_proba)#todo check that all_test_y in bootstrap fold has 2 distinct values for roc
    auc_list_test.append(roc_auc)
    xaxis_test.append([fpr, tpr])
    # ax_roc_merge.plot(fpr, tpr, color="tab:blue", alpha=0.3, linewidth=1)

    fpr, tpr, thresholds = roc_curve(all_train_y, all_train_proba)
    tprs_train.append(tpr)
    fprs_train.append(fpr)
    roc_auc = auc(fpr, tpr)
    auc_list_train.append(roc_auc)
    xaxis_train.append([fpr, tpr])
    # ax_roc_merge.plot(fpr, tpr, color="tab:purple", alpha=0.3, linewidth=1)

    prec_list_test.append(
        precision_score(all_test_y, (np.array(all_test_proba) > 0.5).astype(int))
    )
    prec_list_train.append(
        precision_score(all_train_y, (np.array(all_train_proba) > 0.5).astype(int))
    )

    out = out_dir / str(i)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_test_y_list).to_pickle(out / "all_test_y_list.pkl")
    pd.DataFrame(all_test_proba_list).to_pickle(out / "all_test_proba_list.pkl")
    pd.DataFrame(all_train_y_list).to_pickle(out / "all_train_y_list.pkl")
    pd.DataFrame(all_train_proba_list).to_pickle(out / "all_train_proba_list.pkl")
    pd.DataFrame(prec_list_test).to_pickle(out / "prec_list_test.pkl")
    pd.DataFrame(prec_list_train).to_pickle(out / "prec_list_train.pkl")
    #print(f"{i}/{tot} done.")


def main(path=None, n_bootstrap=100, n_job=6):
    print(f"loading data... path={path}")
    paths = list(path.glob("**/fold_data/*.json"))
    if len(paths) == 0:
        print("There are no .json files in the fold_data folder.")
        return
    out_dir = paths[0].parent.parent / "pickles"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    fig_roc_merge, ax_roc_merge = plt.subplots()
    for filepath in tqdm(paths):
        with open(filepath, "r") as fp:
            try:
                loo_result = json.load(fp)
            except Exception as e:
                print(e)
                return

            training_size = loo_result["training_shape"][0]
            testing_size = loo_result["testing_shape"][0]
            n_peaks = loo_result["n_peak"]
            n_top = loo_result["n_top"]
            max_sample = loo_result["max_sample"]
            window_size = loo_result["w_size"]
            clf = f"{loo_result['clf']}({loo_result['clf_kernel']})"
            pre_proc = "->".join(loo_result["steps"].split("_"))

            data[filepath.stem] = {
                "y_pred_proba_test": loo_result["y_pred_proba_test"],
                "y_test": loo_result["y_test"],
                "y_pred_proba_train": loo_result["y_pred_proba_train"],
                "y_train": loo_result["y_train"],
                "training_size": training_size,
                "testing_size": testing_size,
                "n_peaks": n_peaks,
                "window_size": window_size,
                "clf": clf,
                "pre_proc": pre_proc,
            }

    print("start bootstrap...")
    pool = Pool(processes=n_job)
    with Manager() as manager:
        auc_list_test = manager.list()
        auc_list_train = manager.list()
        tprs_test = manager.list()
        tprs_train = manager.list()
        fprs_test = manager.list()
        fprs_train = manager.list()
        xaxis_train = manager.list()
        xaxis_test = manager.list()
        for i in range(n_bootstrap):
            bootstrap = np.random.choice(paths, size=len(paths), replace=True)
            #print(bootstrap)
            pool.apply_async(
                worker,
                (
                    out_dir,
                    data,
                    i,
                    n_bootstrap,
                    bootstrap,
                    xaxis_train,
                    xaxis_test,
                    auc_list_test,
                    auc_list_train,
                    tprs_test,
                    tprs_train,
                    fprs_test,
                    fprs_train,
                ),
            )
        pool.close()
        pool.join()
        pool.terminate()
        #print("pool done.")
        xaxis_train = list(xaxis_train)
        xaxis_test = list(xaxis_test)
        auc_list_test = list(auc_list_test)
        auc_list_train = list(auc_list_train)

    all_test_y_list = []
    all_test_proba_list = []
    all_train_y_list = []
    all_train_proba_list = []
    prec_list_test = []
    prec_list_train = []
    for i in range(n_bootstrap):
        all_test_y_list.append(pd.read_pickle(out_dir / str(i) / "all_test_y_list.pkl").values.flatten())
        all_test_proba_list.append(pd.read_pickle(out_dir / str(i) / "all_test_proba_list.pkl").values.flatten())
        all_train_y_list.append(pd.read_pickle(out_dir / str(i) / "all_train_y_list.pkl").values.flatten())
        all_train_proba_list.append(pd.read_pickle(out_dir / str(i) / "all_train_proba_list.pkl").values.flatten())
        prec_list_test.append(pd.read_pickle(out_dir / str(i) / "prec_list_test.pkl").values.flatten())
        prec_list_train.append(pd.read_pickle(out_dir / str(i) / "prec_list_train.pkl").values.flatten())

    # prec_list_test = np.mean(prec_list_test)
    # prec_list_train = np.mean(prec_list_train)

    print("building roc...")
    median_auc_test = np.nanmedian(auc_list_test)
    lo_test_auc, hi_test_auc = ninefive_confidence_interval(auc_list_test)
    print(
        f"Testing AUC = {median_auc_test:.2f}({lo_test_auc:.1f}, {hi_test_auc:.1f})"
    )

    median_auc_train = np.nanmedian(auc_list_train)
    lo_train_auc, hi_train_auc = ninefive_confidence_interval(auc_list_train)
    print(
        f"Training AUC = {median_auc_train:.2f}({lo_train_auc:.1f}, {hi_train_auc:.1f})"
    )

    median_prec_test = np.nanmedian(prec_list_test)
    lo_test_prec, hi_test_prec = ninefive_confidence_interval(prec_list_test)
    print(
        f"Testing prec = {median_prec_test:.2f}({lo_test_prec:.1f}, {hi_test_prec:.1f})"
    )

    median_prec_train = np.nanmedian(prec_list_train)
    lo_train_prec, hi_train_prec = ninefive_confidence_interval(prec_list_train)
    print(
        f"Training prec = {median_prec_train:.2f}({lo_train_prec:.1f}, {hi_train_prec:.1f})"
    )

    try:
        xaxis_train_ = random.sample(xaxis_train, 10)
    except ValueError as e:
        print(e)
        xaxis_train_ = xaxis_train

    for fpr, tpr in xaxis_train_:
        ax_roc_merge.plot(fpr, tpr, color="tab:purple", alpha=0.3, linewidth=1)

    xaxis_test_ = random.sample(xaxis_test, 10)
    for fpr, tpr in xaxis_test_:
        ax_roc_merge.plot(fpr, tpr, color="tab:blue", alpha=0.3, linewidth=1)

    ax_roc_merge.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    label = f"Testing (Median AUC = {median_auc_test:.2f}({lo_test_auc:.1f}, {hi_test_auc:.1f})"

    mean_fpr_test, mean_tpr_test = [], []
    for y_list, proba_list in zip(all_test_y_list, all_test_proba_list):
        mean_fpr, mean_tpr, thresholds = roc_curve(
            y_list, proba_list
        )
        mean_fpr_test.append(mean_fpr)
        mean_tpr_test.append(mean_tpr)

    max_length = max(len(arr) for arr in mean_fpr_test)
    mean_fpr_test = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_fpr_test]
    mean_tpr_test = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_tpr_test]

    mean_fpr_test = np.median(mean_fpr_test, axis=0)
    mean_tpr_test = np.median(mean_tpr_test, axis=0)

    ax_roc_merge.plot(
        mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1
    )
    ax_roc_merge.tick_params(axis='x', labelsize=18)  # Adjust the fontsize as needed for the x-axis
    ax_roc_merge.tick_params(axis='y', labelsize=18)
    ax_roc_merge.set_xlabel("False positive rate", fontsize=22)
    ax_roc_merge.set_ylabel("True positive rate", fontsize=22)
    ax_roc_merge.legend(loc="lower right", fontsize=14)
    # fig.show()

    label = f"Training (Median AUC = {median_auc_train:.2f}({lo_train_auc:.1f}, {hi_train_auc:.1f})"
    mean_fpr_train, mean_tpr_train = [], []
    for y_list, proba_list in zip(all_train_y_list, all_train_proba_list):
        mean_fpr, mean_tpr, thresholds = roc_curve(
            y_list, proba_list
        )
        mean_fpr_train.append(mean_fpr)
        mean_tpr_train.append(mean_tpr)

    max_length = max(len(arr) for arr in mean_fpr_train)
    mean_fpr_train = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_fpr_train]
    mean_tpr_train = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_tpr_train]

    mean_fpr_train = np.median(mean_fpr_train, axis=0)
    mean_tpr_train = np.median(mean_tpr_train, axis=0)

    ax_roc_merge.plot(
        mean_fpr_train, mean_tpr_train, color="red", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        #title=f"Receiver operating characteristic (n_bootstrap={n_bootstrap})",
    )
    ax_roc_merge.legend(loc="lower right")

    fig_roc_merge.tight_layout()
    path_ = path / "roc_curve"
    path_.mkdir(parents=True, exist_ok=True)
    # final_path = path / f"{tag}_roc_{classifier_name}.png"
    # print(final_path)
    # fig.savefig(final_path)

    final_path = (
        path_
        / f"{n_bootstrap}_{max_sample}_{n_top}_{window_size}_{n_peaks}_{loo_result['steps']}.png"
    )
    print(final_path)

    fig_roc_merge.set_size_inches(6, 6)
    fig_roc_merge.tight_layout()
    fig_roc_merge.savefig(final_path, dpi=500)

    return [
        f"{median_auc_test:.2f} ({lo_test_auc:.2f}-{hi_test_auc:.2f})",
        f"{median_auc_train:.2f} ({lo_train_auc:.2f}-{hi_train_auc:.2f})",
        f"{median_prec_test:.2f} ({lo_test_prec:.2f}-{hi_test_prec:.2f})",
        f"{median_prec_train:.2f} ({lo_train_prec:.2f}-{hi_train_prec:.2f})",
        training_size,
        testing_size,
        n_peaks,
        max_sample,
        n_top,
        window_size,
        clf,
        pre_proc,
        median_auc_test,
        median_auc_train,
        auc_list_test,
        auc_list_train,
        paths,
        get_time_of_day(path)
    ]


def bootstrap_roc():
    out_dir = Path("E:/Cats/output_test3")
    results = []
    folders = list(out_dir.glob("**/fold_data"))
    for i, item in enumerate(folders):
        # if "1000__005__0_00100__030\cats_LeaveOneOut_-1_-1_QN_rbf" not in str(item):
        #     continue
        print(item)
        print(f"bootstrap_roc {i}/{len(folders)}...")

        res = main(item, n_bootstrap=10)
        if res is not None:
            results.append(res)
    return results


def boostrap_auc_peak_delta(results, out_dir):
    if len(results) == 0:
        print("None value in results!")
        return

    df = pd.DataFrame(
        results,
        columns=[
            "AUC testing (95% CI)",
            "AUC training (95% CI)",
            "Class1 Precision testing (95% CI)",
            "Class1 Precision training (95% CI)",
            "N training samples",
            "N testing samples",
            "N peaks",
            "Max sample count per indiv",
            "N top",
            "Sample length (seconds)",
            "Classifier",
            "Pre-processing",
            "median_auc_test",
            "median_auc_train",
            "median_auc_test_bootstrap",
            "median_auc_train_bootstrap",
            "path",
            "time_of_day"
        ],
    )

    df = df[~pd.isna(df["median_auc_test"])]
    df.loc[
        df["median_auc_test"] <= 0.5, "median_auc_test"
    ] = 0.5  # skitlearn can output auc <0.5 those are chance
    # df = df[df["N peaks"] < 6]
    df = df.sort_values("N peaks", ascending=True)

    df["pipeline"] = df["Pre-processing"] + "->" + df["Classifier"]
    df["N peaks"] = df["N peaks"].astype(int)

    n_peak = df["N peaks"].astype(int).max()
    x_axis = np.arange(1, n_peak+1).astype(str)

    print(df)

    dfs_ntop = [group for _, group in df.groupby(["N top", "time_of_day", "Max sample count per indiv", "Sample length (seconds)"])]
    for df in dfs_ntop:
        ntop = int(df["N top"].values[0])
        s_length = df["Sample length (seconds)"].values[0]
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        dfs = [group for _, group in df.groupby(["pipeline"])]

        x_ticks = sorted(df["N peaks"].unique())
        y_ticks = []
        for x_t in x_ticks:
            y_ = df[df["N peaks"] == x_t][["N training samples", "N testing samples"]].values[0].sum()
            y_ticks.append(y_)

        ax2.bar(
            x_ticks,
            y_ticks,
            color="grey",
            label="n samples",
            alpha=0.4,
            width=0.2,
        )

        colors = list(mcolors.TABLEAU_COLORS.keys())
        print(colors)
        cpt = 0
        colors_ = []
        label_ = []

        #find normalised pipeline for substraction
        df_l1 = pd.DataFrame()

        for val in dfs:
            df_l1["median_auc_test"] = [0] * len(val)
            if val["pipeline"].values[0] == "L1->SVC(rbf)":
                df_l1 = val
                break

        for i, item in enumerate(dfs):
            dfs_ = [group for _, group in item.groupby(["Sample length (seconds)"])]
            for df_ in dfs_:
                print(df_["pipeline"].tolist()[0])
                label = f"Sample length={df_['Sample length (seconds)'].tolist()[0]}s | {'>'.join(df_['pipeline'].tolist()[0].split('_')[:])}"
                ax1.plot(
                    df_["N peaks"],
                    df_l1["median_auc_test"].values - df_["median_auc_test"].values,
                    label=label,
                    marker="x",
                    color=colors[cpt],
                )

                intervals = pd.eval(df_["median_auc_test_bootstrap"].astype(str)) #todo fix strange bug eval should not be needed, seems that there is a mix of types in the column
                if "median_auc_test_bootstrap" in df_l1.columns:
                    intervals_l1 = pd.eval(df_l1["median_auc_test_bootstrap"].astype(str))
                    intervals_diff = np.array(intervals) - np.array(intervals_l1)
                    intervals = [arr.tolist() for arr in intervals_diff]

                perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
                top = perct[2, :]
                bottom = perct[0, :]
                x = df_["N peaks"].values
                ax1.fill_between(
                    x, top.astype(float), bottom.astype(float), alpha=0.1, color=colors[cpt]
                )

                ax1.plot(
                    df_["N peaks"],
                    df_l1["median_auc_train"].values - df_["median_auc_train"].values,
                    # label=f"Train Window size={df_['window_size_list'].tolist()[0]*2} sec | {'>'.join(df_['p_steps_list'].tolist()[0].split('_')[4:])}",
                    marker="s",
                    linestyle="-.",
                    color=colors[cpt],
                )

                intervals = pd.eval(df_["median_auc_train_bootstrap"].astype(str)) #same bug here
                perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
                top = perct[2, :]
                bottom = perct[0, :]
                x = df_["N peaks"].values
                ax1.fill_between(
                    x, top.astype(float), bottom.astype(float), alpha=0.1, color=colors[cpt]
                )

                colors_.append(colors[cpt])
                label_.append(label)

                cpt += 1
                if cpt >= len(colors):
                    cpt = 0

        x_labels = [int(x) if isinstance(x, int) or x.is_integer() else ' ' for x in ax1.get_xticks()]

        ax1.set_xticklabels(x_labels)

        ax1.axhline(y=0.5, color="black", linestyle="--")
        fig.suptitle("Delta L1 Normalised AUC with N peak increase")
        ax1.set_xlabel("Number of peaks")
        ax1.set_ylabel("Delta AUC (Compared to Normalised)")
        ax2.set_ylabel("Number of samples")
        # plt.legend()
        # ax1.legend(loc="lower right").set_visible(True)
        ax2.legend(loc="upper left").set_visible(True)

        color_data = []
        for item in colors_:
            color_data.append((item, "--"))

        ax1.legend(
            color_data, label_, loc="lower right", handler_map={tuple: AnyObjectHandler()}
        )
        ax1.grid()

        ax1.set_ylim(-0.4, 0.25)

        time_of_day = df["time_of_day"].values[0]
        max_scount = df["Max sample count per indiv"].values[0]
        filename = f"{time_of_day}_{ntop}_{max_scount}_{s_length}_auc_per_npeak_bootstrap_delta.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename
        print(filepath)
        # fig.savefig(filepath)
        fig.set_size_inches(6, 4)
        fig.tight_layout()
        fig.savefig(filepath, dpi=500)


def boostrap_auc_peak(results, out_dir):
    if len(results) == 0:
        print("None value in results!")
        return

    df = pd.DataFrame(
        results,
        columns=[
            "AUC testing (95% CI)",
            "AUC training (95% CI)",
            "Class1 Precision testing (95% CI)",
            "Class1 Precision training (95% CI)",
            "N training samples",
            "N testing samples",
            "N peaks",
            "Max sample count per indiv",
            "N top",
            "Sample length (seconds)",
            "Classifier",
            "Pre-processing",
            "median_auc_test",
            "median_auc_train",
            "median_auc_test_bootstrap",
            "median_auc_train_bootstrap",
            "path",
            "time_of_day"
        ],
    )

    df = df[~pd.isna(df["median_auc_test"])]
    df.loc[
        df["median_auc_test"] <= 0.5, "median_auc_test"
    ] = 0.5  # skitlearn can output auc <0.5 those are chance
    # df = df[df["N peaks"] < 6]
    df = df.sort_values("N peaks", ascending=True)

    df["pipeline"] = df["Pre-processing"] + "->" + df["Classifier"]
    df["N peaks"] = df["N peaks"].astype(int)

    n_peak = df["N peaks"].astype(int).max()
    x_axis = np.arange(1, n_peak+1).astype(str)

    df.to_csv(out_dir / "results.csv", index=False)
    print(df)

    dfs_ntop = [group for _, group in df.groupby(["N top", "time_of_day", "Max sample count per indiv", "Sample length (seconds)", "Classifier"])]
    for df in dfs_ntop:
        ntop = int(df["N top"].values[0])
        s_length = df["Sample length (seconds)"].values[0]
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        dfs = [group for _, group in df.groupby(["pipeline"])]

        x_ticks = sorted(df["N peaks"].unique())
        y_ticks = []
        for x_t in x_ticks:
            y_ = df[df["N peaks"] == x_t][["N training samples", "N testing samples"]].values[0].sum()
            y_ticks.append(y_)

        ax2.bar(
            x_ticks,
            y_ticks,
            color="grey",
            label="n samples",
            alpha=0.4,
            width=0.2,
        )

        colors = list(mcolors.TABLEAU_COLORS.keys())
        print(colors)
        cpt = 0
        colors_ = []
        label_ = []

        for i, item in enumerate(dfs):
            dfs_ = [group for _, group in item.groupby(["Sample length (seconds)"])]
            for df_ in dfs_:
                print(df_["pipeline"].tolist()[0])
                label = f"Sample length={df_['Sample length (seconds)'].tolist()[0]}s | {'>'.join(df_['pipeline'].tolist()[0].split('_')[:])}"
                ax1.plot(
                    df_["N peaks"],
                    df_["median_auc_test"],
                    label=label,
                    marker="x",
                    color=colors[cpt],
                )

                intervals = pd.eval(df_["median_auc_test_bootstrap"].astype(str)) #todo fix strange bug eval should not be needed, seems that there is a mix of types in the column
                perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
                top = perct[2, :]
                bottom = perct[0, :]
                x = df_["N peaks"].values
                ax1.fill_between(
                    x, top.astype(float), bottom.astype(float), alpha=0.1, color=colors[cpt]
                )

                ax1.plot(
                    df_["N peaks"],
                    df_["median_auc_train"],
                    # label=f"Train Window size={df_['window_size_list'].tolist()[0]*2} sec | {'>'.join(df_['p_steps_list'].tolist()[0].split('_')[4:])}",
                    marker="s",
                    linestyle="-.",
                    color=colors[cpt],
                )

                intervals = pd.eval(df_["median_auc_train_bootstrap"].astype(str)) #same bug here
                perct = np.percentile(intervals, [2.5, 50, 97.5], axis=1)
                top = perct[2, :]
                bottom = perct[0, :]
                x = df_["N peaks"].values
                ax1.fill_between(
                    x, top.astype(float), bottom.astype(float), alpha=0.1, color=colors[cpt]
                )

                colors_.append(colors[cpt])
                label_.append(label)

                cpt += 1
                if cpt >= len(colors):
                    cpt = 0

        x_labels = [int(x) if isinstance(x, int) or x.is_integer() else ' ' for x in ax1.get_xticks()]

        ax1.set_xticklabels(x_labels)

        ax1.axhline(y=0.5, color="black", linestyle="--")
        fig.suptitle("Evolution of AUC(training and testing) with N peak increase")
        ax1.set_xlabel("Number of peaks")
        ax1.set_ylabel("Median AUC")
        ax2.set_ylabel("Number of samples")
        # plt.legend()
        # ax1.legend(loc="lower right").set_visible(True)
        ax2.legend(loc="upper left").set_visible(True)

        color_data = []
        for item in colors_:
            color_data.append((item, "--"))

        ax1.legend(
            color_data, label_, loc="lower right", handler_map={tuple: AnyObjectHandler()}
        )
        ax1.grid()

        ax1.set_ylim(0.45, 1)

        time_of_day = df["time_of_day"].values[0]
        max_scount = df["Max sample count per indiv"].values[0]
        clf = df["Classifier"].values[0].replace("(", "_").replace(")", "_")
        filename = f"{clf}_{time_of_day}_{ntop}_{max_scount}_{s_length}_auc_per_npeak_bootstrap.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename
        print(filepath)
        # fig.savefig(filepath)
        fig.set_size_inches(6, 4)
        fig.tight_layout()
        fig.savefig(filepath, dpi=500)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        res_folder = Path(sys.argv[1])
        n_bootstrap = int(sys.argv[2])
        n_job = int(sys.argv[3])
    else:
        res_folder = Path("E:/Cats/paper_debug_regularisation_8/")
        n_bootstrap = 20
        n_job = 2

    results = []
    folders = [
        x
        for x in res_folder.glob("*/*/*")
        if x.is_dir()
    ]
    for i, item in enumerate(folders):
        print(f"{i}/{len(folders)}...")
        print(item)
        res = main(item, n_bootstrap=n_bootstrap, n_job=n_job)
        #auc, optimal_threshold, optimal_sensitivity, optimal_specificity = eval_recall(Path(f"{item}/fold_data"))
        if res is not None:
            results.append(res)

    boostrap_auc_peak_delta(results, res_folder)

    df = pd.DataFrame(
        results,
        columns=[
            "AUC testing (95% CI)",
            "AUC training (95% CI)",
            "Class1 Precision testing (95% CI)",
            "Class1 Precision training (95% CI)",
            "N training samples",
            "N testing samples",
            "N peaks",
            "Max samples",
            "n Top",
            "Sample length (seconds)",
            "Classifier",
            "Pre-processing",
            "median_auc_test",
            "median_auc_train",
            "auc_list_test",
            "auc_list_train",
            "path",
            "time_of_day"
        ],
    )
    df = df.sort_values("median_auc_test", ascending=False)
    df_ = df.sort_values("median_auc_test", ascending=False)
    df_ = df_.drop("median_auc_test", axis=1)
    df_ = df_.drop("path", axis=1)
    df_ = df_.drop("time_of_day", axis=1)
    df_ = df_.drop("auc_list_train", axis=1)
    df_ = df_.drop("auc_list_test", axis=1)
    df_ = df_.drop("n Top", axis=1)
    df_ = df_.drop("Max samples", axis=1)
    df_ = df_.head(20)
    print(df_.to_latex(index=False))
    df.to_csv(res_folder / "cat_result_table.csv", index=False)

    df_noproc = df[df["Pre-processing"] == '']
    df_noproc = df_noproc.sort_values("N peaks")
    df_noproc_auc = df_noproc["median_auc_test"].values

    df_l1 = df[df["Pre-processing"] == 'L1']
    df_l1 = df_l1.sort_values("N peaks")
    df_l1_auc = df_l1["median_auc_test"].values

    # First, conduct the Wilcoxon signed-rank test
    print("df_l1_auc", df_l1_auc)
    print("df_noproc_auc", df_noproc_auc)
    wilcoxon_p_value = stats.wilcoxon(df_l1_auc - df_noproc_auc, alternative='less').pvalue
    print(f"Wilcoxon Signed-Rank Test: p-value = {wilcoxon_p_value}")

    # t_stat, p_value = ttest_rel(df_l1_auc, df_noproc_auc)
    # print(f"Paired T-Test: t-statistic = {t_stat}, p-value = {p_value}")