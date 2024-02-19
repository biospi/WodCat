import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_heatmap(df, col, out_dir, title=""):
    scores = df[col].values
    scores = np.array(scores).reshape(len(df["C"].unique()), len(df["gamma"].unique()))
    #plt.figure(figsize=(8, 6))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    fig, ax = plt.subplots()
    im = ax.imshow(scores[::-1, :], interpolation='nearest')
    # im = ax.imshow(scores, interpolation='nearest',
    #            norm=MidpointNormalize(vmin=-.2, midpoint=0.5))
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    fig.colorbar(im)
    ax.set_xticks(np.arange(len(df["gamma"].unique())),
               [np.format_float_scientific(i, 1) for i in df["gamma"].unique()], rotation=45)
    ax.set_yticks(np.arange(len(df["C"].unique()))[::-1],
               [np.format_float_scientific(i, ) for i in df["C"].unique()])
    ax.set_title(f'Regularisation Accuracy\n{title}')
    fig.tight_layout()
    fig.show()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"heatmap_{col}_{title}.png".replace(":", "_").replace(" ", "_")
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


def plot_fig(df, col, out_dir, title=""):
    scores = df[col].values
    scores = np.array(scores).reshape(len(df["C"].unique()), len(df["gamma"].unique()))
    Cs = df["C"].unique()
    Gammas = df["gamma"].unique()
    fig, ax = plt.subplots()
    for ind, i in enumerate(Cs):
        ax.plot(Gammas, scores[ind], label="C: " + str(i))
    ax.set_title(title)
    ax.set_xscale('log')
    ax.legend()
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.show()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"plot_{col}_{title}.png".replace(":", "_").replace(" ", "_")
    filepath = out_dir / filename
    print(filepath)
    fig.savefig(filepath)


def regularisation_heatmap(data_dir, out_dir):
    files = list(data_dir.glob("*.csv"))
    dfs = []
    mean_test_score_list = []
    mean_train_score_list = []
    for file in files:
        df = pd.read_csv(file)
        data = df[["param_gamma", "param_C", "mean_test_score", "mean_train_score"]]
        data = data.assign(fold=int(file.stem.split('_')[1]))
        data = data.sort_values(["param_gamma", "param_C"])
        mean_test_score_list.append(data["mean_test_score"].values)
        mean_train_score_list.append(data["mean_train_score"].values)
        dfs.append(data)
    df_data = pd.DataFrame()
    df_data["gamma"] = df["param_gamma"]
    df_data["C"] = df["param_C"]
    df_data["mean_test_score"] = pd.DataFrame(mean_test_score_list).mean()
    df_data["mean_train_score"] = pd.DataFrame(mean_train_score_list).mean()

    plot_heatmap(
        df_data,
        "mean_train_score",
        out_dir,
        f"GridSearch Training model:{data_dir.parent.parent.name}",
    )

    plot_heatmap(
        df_data,
        "mean_test_score",
        out_dir,
        f"GridSearch Testing model:{data_dir.parent.parent.name}",
    )

    # plot_fig(
    #     df_data,
    #     "mean_test_score",
    #     out_dir,
    #     f"GridSearch Testing model:{data_dir.parent.parent.name}",
    # )


if __name__ == "__main__":
    input_folder = Path("E:/Cats/paper/All_50_10_030_001/rbf/QN_LeaveOneOut/models/GridSearchCV_rbf_QN")
    out_dir = Path("E:/Cats/paper/All_50_10_030_001/rbf/_LeaveOneOut")
    regularisation_heatmap(input_folder, out_dir)