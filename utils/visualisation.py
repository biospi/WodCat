import pathlib
import random
from collections import Counter
from datetime import datetime, timedelta
from operator import itemgetter
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
# import umap
# import umap.plot
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme, element_text
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import auc
from sklearn.model_selection import learning_curve
from tqdm import tqdm

from cwt._cwt import CWT, plot_line, STFT, DWT
from highdimensional.decisionboundaryplot import DBPlot
from utils._anscombe import anscombe
from utils._normalisation import CenterScaler
from utils.utils import concatenate_images, time_of_day_


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    return date_list


def add_separator(df_):
    df_ = df_.reset_index(drop=True)
    idxs = []
    d = df_["animal_ids"].values
    for i in range(d.size - 1):
        if d[i] != d[i + 1]:
            idxs.append(i)
    df_ = df_.reindex(df_.index.values.tolist() + [str(x).zfill(5) + "a" for x in idxs])
    df_.index = [str(x).zfill(5) for x in df_.index]
    df_ = df_.sort_index()
    ni = (
        pd.Series(df_["animal_ids"].astype(float).values)
        .interpolate(method="nearest")
        .values
    )
    df_["animal_ids"] = ni.tolist()
    nt = (
        pd.Series(df_["target"].astype(float).values)
        .interpolate(method="nearest")
        .values
    )
    df_["target"] = nt.astype(int).tolist()
    return df_


def plot_crepuscular(out_dir, df, filename="median_peak", ylabel="Activity count", n_xtick=6):
    out = out_dir / "crepuscular"
    out.mkdir(parents=True, exist_ok=True)

    if "peak0_datetime" not in df.columns:
        return
    df["peak0_datetime"] = pd.to_datetime(df["peak0_datetime"], format="'%Y-%m-%d%H:%M:%S'")
    df['hour'] = df["peak0_datetime"].dt.hour
    df['time_of_day'] = df['hour'].apply(time_of_day_)

    colors = ['#FFD700', '#FFA07A', '#98FB98', '#FFB6C1', '#ADD8E6']
    unique_times = ['Day', 'Night']
    fig, axs = plt.subplots(1, len(unique_times), figsize=(10, 4), sharey=True)  # Adjust figsize as needed
    for idx, (item, color) in enumerate(zip(unique_times, colors)):
        df_ = df[df["time_of_day"] == item]
        df_activity = df_[[c for c in df_.columns if c.isdigit()]]
        median_curve = df_activity.median(axis=0)
        lower_bound = df_activity.quantile(0.025, axis=0)
        upper_bound = df_activity.quantile(0.975, axis=0)
        ax = axs[idx]
        ax.plot(median_curve, label='Median peak', color='black')
        ax.fill_between(df_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=color,
                        label='Spread (95th percentile)')
        ax.set_xlabel('Time in seconds')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{item}({len(df_)})')
        if idx == 0:
            ax.legend()
        ax.grid(True)

        x_ticks = np.linspace(0, len(median_curve) - 1, n_xtick, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(median_curve.index[i])) for i in x_ticks])

        fig_, ax_ = plt.subplots()
        ax_.plot(median_curve, label='Median peak', color='black')
        ax_.fill_between(df_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=color, label='Spread(95th percentile)')
        ax_.set_xlabel('Time in seconds')
        ax_.set_ylabel(ylabel)
        ax_.set_title(f'Median peak ({len(df_activity)}) with spread(95th percentile ) at {item}')
        ax_.legend()
        ax_.grid(True)
        x_ticks = np.linspace(0, len(median_curve) - 1, n_xtick, dtype=int)
        ax_.set_xticks(x_ticks)
        ax_.set_xticklabels([str(int(median_curve.index[i])) for i in x_ticks])
        filepath = out / f'{filename}_{item}.png'.replace('/', '_')
        fig_.savefig(filepath, bbox_inches='tight')

    fig.tight_layout()
    filepath = out / f'{filename}.png'
    print(filepath)
    fig.savefig(filepath, bbox_inches='tight')


def plot_groups(
    N_META,
    animal_ids,
    class_healthy_label,
    class_unhealthy_label,
    graph_outputdir,
    df,
    title="title",
    xlabel="xlabel",
    ylabel="target",
    ntraces=1,
    idx_healthy=None,
    idx_unhealthy=None,
    show_max=True,
    show_min=False,
    show_mean=True,
    show_median=True,
    stepid=0,
):
    """Plot all rows in dataframe for each class Health or Unhealthy.

    Keyword arguments:
    df -- input dataframe containing samples (activity data, label/target)
    """
    df_healthy = df[df["health"] == 0].iloc[:, :-N_META].values
    df_unhealthy = df[df["health"] == 1].iloc[:, :-N_META].values

    assert len(df_healthy) > 0, "no healthy samples!"
    assert len(df_unhealthy) > 0, "no unhealthy samples!"

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.80, 4.20))
    fig.suptitle(title, fontsize=18)

    ymin = np.min(df.iloc[:, :-N_META].values)
    if idx_healthy is None or idx_unhealthy is None:
        ymax = np.max(df.iloc[:, :-N_META].values)
    else:
        ymax = max(
            [np.max(df_healthy[idx_healthy]), np.max(df_unhealthy[idx_unhealthy])]
        )

    if show_max:
        ymax = np.max(df_healthy)

    ticks = get_time_ticks(df_healthy.shape[1])

    if idx_healthy is None and ntraces is not None:
        idx_healthy = random.sample(range(1, df_healthy.shape[0]), ntraces)
    if ntraces is None:
        idx_healthy = list(range(df_healthy.shape[0]))
        idx_unhealthy = list(range(df_unhealthy.shape[0]))

    for i in idx_healthy:
        ax1.plot(ticks, df_healthy[i])
        ax1.set(xlabel=xlabel, ylabel=ylabel)
        if ntraces is None:
            ax1.set_title(
                "Healthy(%s) animals %d / displaying %d"
                % (class_healthy_label, df_healthy.shape[0], df_healthy.shape[0])
            )
        else:
            ax1.set_title(
                "Healthy(%s) animals %d / displaying %d"
                % (class_healthy_label, df_healthy.shape[0], ntraces)
            )
        ax1.set_ylim([ymin, ymax])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
    if idx_unhealthy is None:
        idx_unhealthy = random.sample(range(1, df_unhealthy.shape[0]), ntraces)
    for i in idx_unhealthy:
        ax2.plot(ticks, df_unhealthy[i])
        ax2.set(xlabel=xlabel, ylabel=ylabel)
        ax2.set_xticklabels(ticks, fontsize=12)
        if ntraces is None:
            ax2.set_title(
                "Unhealthy(%s) %d samples / displaying %d"
                % (class_unhealthy_label, df_unhealthy.shape[0], df_unhealthy.shape[0])
            )
        else:
            ax2.set_title(
                "Unhealthy(%s) animals %d / displaying %d"
                % (class_unhealthy_label, df_unhealthy.shape[0], ntraces)
            )
        ax2.set_ylim([ymin, ymax])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
    if show_max:
        # ax1.plot(ticks, np.amax(df_healthy, axis=0), c='tab:gray', label='max', linestyle='-')
        # ax2.plot(ticks, np.amax(df_unhealthy, axis=0), c='tab:gray', label='max', linestyle='-')
        ax1.fill_between(
            ticks,
            np.amax(df_healthy, axis=0),
            color="lightgrey",
            label="max",
            zorder=-1,
        )
        ax2.fill_between(
            ticks, np.amax(df_unhealthy, axis=0), label="max", color="lightgrey"
        )
        ax1.legend()
        ax2.legend()
    if show_min:
        ax1.plot(ticks, np.amin(df_healthy, axis=0), c="red", label="min")
        ax2.plot(ticks, np.amin(df_unhealthy, axis=0), c="red", label="min")
        ax1.legend()
        ax2.legend()

    if show_mean:
        ax1.plot(
            ticks,
            np.mean(df_healthy, axis=0),
            c="black",
            label="mean",
            alpha=1,
            linestyle="-",
        )
        ax2.plot(
            ticks,
            np.mean(df_unhealthy, axis=0),
            c="black",
            label="mean",
            alpha=1,
            linestyle="-",
        )
        ax1.legend()
        ax2.legend()

    if show_median:
        ax1.plot(
            ticks,
            np.median(df_healthy, axis=0),
            c="black",
            label="median",
            alpha=1,
            linestyle=":",
        )
        ax2.plot(
            ticks,
            np.median(df_unhealthy, axis=0),
            c="black",
            label="median",
            alpha=1,
            linestyle=":",
        )
        ax1.legend()
        ax2.legend()

    # plt.show()
    fig.tight_layout()
    filename = f"{stepid}_{title.replace(' ', '_')}.png"
    filepath = graph_outputdir / filename
    # print('saving fig...')
    fig.savefig(filepath)

    print("building heatmaps...")
    cbarlocs = [0.81, 0.19]
    # add row separator
    df_ = df.copy()
    df_["animal_ids"] = animal_ids

    df_healthy_ = add_separator(df_[df_["health"] == 0])
    df_unhealthy_ = add_separator(df_[df_["health"] == 1])

    t1 = "Healthy(%s) %d animals  %d samples" % (
        class_healthy_label,
        df_healthy_["animal_ids"].astype(str).drop_duplicates().size,
        df_healthy_.shape[0],
    )
    t2 = "UnHealthy(%s) %d animals %d samples" % (
        class_unhealthy_label,
        df_unhealthy_["animal_ids"].astype(str).drop_duplicates().size,
        df_unhealthy_.shape[0],
    )
    fig_ = make_subplots(
        rows=2, cols=1, x_title=xlabel, y_title="Transponder", subplot_titles=(t1, t2)
    )
    fig_.add_trace(
        go.Heatmap(
            z=df_healthy_.iloc[:, :-2],
            x=ticks,
            y=[
                str(int(float(x[0]))) + "_" + str(x[1])
                for x in zip(
                    df_healthy_["animal_ids"].astype(str).tolist(),
                    list(range(df_healthy_.shape[0])),
                )
            ],
            colorbar=dict(len=0.40, y=cbarlocs[0]),
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )

    fig_.add_trace(
        go.Heatmap(
            z=df_unhealthy_.iloc[:, :-2],
            x=ticks,
            y=[
                str(int(float(x[0]))) + "_" + str(x[1])
                for x in zip(
                    df_unhealthy_["animal_ids"].astype(str).tolist(),
                    list(range(df_unhealthy_.shape[0])),
                )
            ],
            colorbar=dict(len=0.40, y=cbarlocs[1]),
            colorscale="Viridis",
        ),
        row=2,
        col=1,
    )
    fig_["layout"]["xaxis"]["tickformat"] = "%H:%M"
    fig_["layout"]["xaxis2"]["tickformat"] = "%H:%M"

    zmin = min([np.min(df_unhealthy.flatten()), np.min(df_unhealthy.flatten())])
    zmax = max([np.max(df_unhealthy.flatten()), np.max(df_unhealthy.flatten())])

    fig_.data[0].update(zmin=zmin, zmax=zmax)
    fig_.data[1].update(zmin=zmin, zmax=zmax)

    fig_.update_layout(title_text=title)
    filename = f"{stepid}_{title.replace(' ', '_')}_heatmap.html"
    filepath = graph_outputdir / filename
    print(filepath)
    fig_.write_html(filepath.as_posix())

    fig.clear()
    plt.close(fig)

    return idx_healthy, idx_unhealthy


def plot_2d_space(
    X, y, filename_2d_scatter, label_series, title="title", colors=None, marker_size=4
):
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    print("plot_2d_space")
    if len(X[0]) == 1:
        for l in zip(np.unique(y)):
            ax.scatter(
                X[y == l, 0], np.zeros(X[y == l, 0].size), label=l, s=marker_size
            )
    else:
        for l in zip(np.unique(y)):
            if l[0] in label_series.keys():
                ax.scatter(
                    X[y == l[0]][:, 0],
                    X[y == l[0]][:, 1],
                    label=label_series[l[0]],
                    s=marker_size,
                )

    colormap = cm.get_cmap("Spectral")
    colorst = [colormap(i) for i in np.linspace(0, 0.9, len(ax.collections))]
    for t, j1 in enumerate(ax.collections):
        j1.set_color(colorst[t])
    ax.patch.set_facecolor("black")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=5)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    print(filename_2d_scatter)
    folder = Path(filename_2d_scatter).parent
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename_2d_scatter)
    # plt.show()
    plt.close(fig)
    plt.clf()


# def plot_umap(meta_columns, df, output_dir, label_series, title="title", y_col="label"):
#     pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
#     df_before_reduction = df.iloc[:, : -len(meta_columns)].values
#     embedding = umap.UMAP().fit(df_before_reduction)
#
#     ids = df["id"].values
#     labels = df[y_col].values
#     seasons = (
#         pd.to_datetime(df["date"], format="%d/%m/%Y").dt.month % 12 // 3 + 1
#     ).map({1: "winter", 2: "spring", 3: "summer", 4: "fall"})
#     filename = f"{title.replace(' ', '_')}.png"
#
#     fig, ax = plt.subplots(figsize=(9.00, 9.00))
#     umap.plot.points(embedding, labels=labels, ax=ax, background="black")
#     filepath = output_dir / f"umap_plot_labels_{filename}"
#     fig.savefig(filepath)
#     print(filepath)
#
#     fig, ax = plt.subplots(figsize=(9.00, 9.00))
#     umap.plot.points(embedding, labels=ids, ax=ax, background="black")
#     filepath = output_dir / f"umap_plot_ids_{filename}"
#     fig.savefig(filepath)
#     print(filepath)
#
#     fig, ax = plt.subplots(figsize=(9.00, 9.00))
#     umap.plot.points(embedding, labels=seasons, ax=ax, background="black")
#     filepath = output_dir / f"umap_plot_seasons_{filename}"
#     fig.savefig(filepath)
#     print(filepath)


def plot_time_pca(
    meta_size, df, output_dir, label_series, title="title", y_col="label"
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame(PCA(n_components=2).fit_transform(df.iloc[:, :-meta_size])).values
    y = df["target"].astype(int)
    ##y_label = df_time_domain["label"]
    filename = title.replace(" ", "_")
    filepath = output_dir / filename
    plot_2d_space(X, y, filepath, label_series, title=title)


def plot_time_pls(
    meta_size, df, output_dir, label_series, title="title", y_col="label"
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    y = df["target"].astype(int)
    X = pd.DataFrame(
        PLSRegression(n_components=2).fit_transform(X=df.iloc[:, :-meta_size], y=y)[0]
    ).values

    ##y_label = df_time_domain["label"]
    filename = title.replace(" ", "_")
    filepath = output_dir / filename
    plot_2d_space(X, y, filepath, label_series, title=title)


def plot_time_lda(N_META, df, output_dir, label_series, title="title", y_col="label"):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    y = df["target"].astype(int).values
    X = df.iloc[:, :-N_META].values
    n_components = np.unique(y).size - 1
    X = pd.DataFrame(LDA(n_components=n_components).fit_transform(X, y)).values
    # y = df_time_domain.iloc[:, -1].astype(int)
    filename = title.replace(" ", "_")
    filepath = "%s/%s.png" % (output_dir, filename)
    plot_2d_space(X, y, filepath, label_series, title=title)


def string_array2array(string):
    return [
        float(x)
        for x in string.replace("\n", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "")
        .split(" ")
        if len(x) > 0
    ]


def format_for_box_plot(df):
    print("formatForBoxPlot...")
    dfs = []
    for index, row in df.iterrows():
        data = pd.DataFrame()
        test_balanced_accuracy_score = string_array2array(
            row["test_balanced_accuracy_score"]
        )
        test_precision_score0 = string_array2array(row["test_precision_score0"])
        test_precision_score1 = string_array2array(row["test_precision_score1"])
        test_recall_score0 = string_array2array(row["test_recall_score0"])
        test_recall_score1 = string_array2array(row["test_recall_score1"])
        test_f1_score0 = string_array2array(row["test_f1_score0"])
        test_f1_score1 = string_array2array(row["test_f1_score1"])
        roc_auc_scores = string_array2array(row["roc_auc_scores"])
        config = [
            row["config"].replace("->", ">").replace(" ", "")
            for _ in range(len(test_balanced_accuracy_score))
        ]
        data["test_balanced_accuracy_score"] = test_balanced_accuracy_score
        data["test_precision_score0"] = test_precision_score0
        data["test_precision_score1"] = test_precision_score1
        data["test_recall_score0"] = test_recall_score0
        data["test_recall_score1"] = test_recall_score1
        data["test_f1_score0"] = test_f1_score0
        data["test_f1_score1"] = test_f1_score1
        data["class0"] = row["class0"]
        data["class1"] = row["class1"]
        roc_auc_scores.extend(
            [0] * (len(test_balanced_accuracy_score) - len(roc_auc_scores))
        )  # in case auc could not be computed for fold
        data["roc_auc_scores"] = roc_auc_scores
        data["config"] = config
        dfs.append(data)
    formated = pd.concat(dfs, axis=0)
    return formated


def plot_ml_report(clf_name, path, output_dir):
    print("building report visualisation...")
    df = pd.read_csv(str(path), index_col=None)
    medians = []
    for value in df["roc_auc_scores"].values:
        v = string_array2array(value)
        medians.append(np.median(v))
    df["median_auc"] = medians

    df["config"] = f"{df.steps[0]}{df.classifier[0]}"
    df = df.sort_values("median_auc")
    df = df.drop_duplicates(subset=["config"], keep="first")
    df = df.fillna(-1)
    print(df)
    t4 = "AUC performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
        df["days"].values[0],
        df["class0"].values[0],
        df["class_0_label"].values[0],
        df["class1"].values[0],
        df["class_1_label"].values[0],
    )

    t3 = (
        "Accuracy performance of different inputs<br>Days=%d class0=%d %s class1=%d %s"
        % (
            df["days"].values[0],
            df["class0"].values[0],
            df["class_0_label"].values[0],
            df["class1"].values[0],
            df["class_1_label"].values[0],
        )
    )

    t1 = (
        "Precision class0 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s"
        % (
            df["days"].values[0],
            df["class0"].values[0],
            df["class_0_label"].values[0],
            df["class1"].values[0],
            df["class_1_label"].values[0],
        )
    )

    t2 = (
        "Precision class1 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s"
        % (
            df["days"].values[0],
            df["class0"].values[0],
            df["class_0_label"].values[0],
            df["class1"].values[0],
            df["class_1_label"].values[0],
        )
    )

    fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))

    df = format_for_box_plot(df)

    fig.append_trace(
        px.box(df, x="config", y="test_precision_score0").data[0], row=1, col=1
    )
    fig.append_trace(
        px.box(df, x="config", y="test_precision_score1").data[0], row=2, col=1
    )
    fig.append_trace(
        px.box(df, x="config", y="test_balanced_accuracy_score").data[0], row=3, col=1
    )
    fig.append_trace(px.box(df, x="config", y="roc_auc_scores").data[0], row=4, col=1)
    fig.update_xaxes(showticklabels=False)  # hide all the xticks
    fig.update_xaxes(showticklabels=True, row=4, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    fig.update_xaxes(showgrid=True, gridwidth=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"ML_performance_{clf_name}.html"
    print(filepath)
    fig.write_html(str(filepath))
    # fig.show()


def plot_zeros_distrib(
    meta_columns,
    label_series,
    data_frame_no_norm,
    graph_outputdir,
    title="Percentage of zeros in activity per sample",
):
    print("plot_zeros_distrib...")
    data = {}
    target_labels = []
    z_prct = []

    for index, row in data_frame_no_norm.iterrows():
        a = row[: -len(meta_columns)].values
        label = label_series[row["target"]]

        target_labels.append(label)
        z_prct.append(np.sum(a == 0) / len(a))

        if label not in data.keys():
            data[label] = a
        else:
            data[label] = np.append(data[label], a)
    distrib = {}
    for key, value in data.items():
        zeros_count = np.sum(value == np.log(anscombe(0))) / len(value)
        lcount = np.sum(
            data_frame_no_norm["target"] == {v: k for k, v in label_series.items()}[key]
        )
        distrib[str(key) + " (%d)" % lcount] = zeros_count

    plt.bar(range(len(distrib)), list(distrib.values()), align="center")
    plt.xticks(range(len(distrib)), list(distrib.keys()))
    plt.title(title)
    plt.xlabel("Famacha samples (number of sample in class)")
    plt.ylabel("Percentage of zero values in samples")
    # plt.show()
    print(distrib)

    df = pd.DataFrame.from_dict({"Percent of zeros": z_prct, "Target": target_labels})
    graph_outputdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(graph_outputdir / "z_prct_data.data")
    g = (
        ggplot(df)  # defining what data to use
        + aes(
            x="Target", y="Percent of zeros", color="Target", shape="Target"
        )  # defining what variable to use
        + geom_jitter()  # defining the type of plot to use
        + stat_summary(geom="crossbar", color="black", width=0.2)
        + theme(
            subplots_adjust={"right": 0.82}, axis_text_x=element_text(angle=90, hjust=1)
        )
    )

    fig = g.draw()
    fig.tight_layout()
    # fig.show()
    filename = f"zero_percent_{title.lower().replace(' ', '_')}.png"
    filepath = graph_outputdir / filename
    # print('saving fig...')
    print(filepath)
    fig.savefig(filepath)
    # print("saved!")
    fig.clear()
    plt.close(fig)


def mean_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def plot_roc_range(
    ax_roc_merge,
    ax,
    tprs_test,
    mean_fpr_test,
    aucs_test,
    tprs_train,
    mean_fpr_train,
    aucs_train,
    out_dir,
    classifier_name,
    fig,
    fig_roc_merge,
    cv_name,
    info="None",
    tag="",
    export_fig_as_pdf=False,
):
    ax_roc_merge.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )
    ax[0].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )
    ax[1].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0
    mean_auc_test = auc(mean_fpr_test, mean_tpr_test)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs_test)

    label = f"Mean ROC Test (Median AUC = {np.median(aucs_test):.2f}, 95% CI [{lo:.4f}, {hi:.4f}] )"
    if len(aucs_test) <= 2:
        label = r"Mean ROC (Median AUC = %0.2f)" % np.median(aucs_test)
    ax[1].plot(mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1)

    ax[1].set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Testing data) Receiver operating characteristic cv:{cv_name} \n info:{info}",
    )
    ax[1].set_xlabel("False positive rate")
    ax[1].set_ylabel("True positive rate")
    ax[1].legend(loc="lower right")

    ax_roc_merge.plot(
        mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training/Testing data) Receiver operating characteristic cv:{cv_name} \n info:{info}",
    )
    ax_roc_merge.set_xlabel("False positive rate")
    ax_roc_merge.set_ylabel("True positive rate")
    ax_roc_merge.legend(loc="lower right")
    # fig.show()

    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0
    mean_auc_train = auc(mean_fpr_train, mean_tpr_train)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs_train)

    label = f"Mean ROC Training (Median AUC = {np.median(aucs_train):.2f}, 95% CI [{lo:.4f}, {hi:.4f}] )"
    if len(aucs_train) <= 2:
        label = r"Mean ROC (Median AUC = %0.2f)" % np.median(aucs_train)
    ax[0].plot(
        mean_fpr_train, mean_tpr_train, color="black", label=label, lw=2, alpha=1
    )

    ax[0].set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training data) Receiver operating characteristic cv:{cv_name} \n info:{info}",
    )
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].legend(loc="lower right")

    ax_roc_merge.plot(
        mean_fpr_train, mean_tpr_train, color="red", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"(Training data) Receiver operating characteristic cv:{cv_name} \n info:{info}",
    )
    ax_roc_merge.legend(loc="lower right")

    fig.tight_layout()
    path = out_dir / "roc_curve" / cv_name
    path.mkdir(parents=True, exist_ok=True)
    # final_path = path / f"{tag}_roc_{classifier_name}.png"
    # print(final_path)
    # fig.savefig(final_path)

    final_path = path / f"{tag}_roc_{classifier_name}_merge.png"
    print(final_path)
    fig_roc_merge.savefig(final_path)

    filepath = out_dir.parent / f"{out_dir.stem}_{tag}_roc_{classifier_name}_merge.png"
    print(filepath)
    fig_roc_merge.savefig(filepath)

    if export_fig_as_pdf:
        final_path = path / f"{tag}_roc_{classifier_name}_merge.pdf"
        print(final_path)
        fig_roc_merge.savefig(final_path)

        filepath = (
            out_dir.parent / f"{out_dir.stem}_{tag}_roc_{classifier_name}_merge.pdf"
        )
        print(filepath)
        fig_roc_merge.savefig(filepath)
    return mean_auc_test


def plot_mean_groups(
    sub_sample_scales,
    n_scales,
    sfft_window,
    wavelet_f0,
    dwt_w,
    df,
    label_series,
    N_META,
    out_dir,
    filename="mean_of_groups.html",
):
    print("plot mean group...")
    traces = []
    fig_group_means = go.Figure()
    fig_group_median = go.Figure()
    for key in tqdm(label_series.keys()):
        df_ = df[df["target"] == key]
        if df_.shape[0] == 0:
            continue
        fig_group = go.Figure()
        n = df_.shape[0]
        for index, row in df_.iterrows():
            x = np.arange(row.shape[0] - N_META)
            y = row.iloc[:-N_META].values
            id = str(int(float(row.iloc[-4])))
            date = row.iloc[-2]
            label = label_series[key]
            name = "%s %s %s" % (id, date, label)
            fig_group.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        mean = np.mean(df_.iloc[:, :-N_META], axis=0)
        median = np.median(df_.iloc[:, :-N_META], axis=0)

        try:
            dfs_mean = [
                (g["name"].values[0], np.mean(g.iloc[:, :-N_META], axis=0))
                for _, g in df_.groupby(["name"])
            ]
            dfs_median = [
                (g["name"].values[0], np.median(g.iloc[:, :-N_META], axis=0))
                for _, g in df_.groupby(["name"])
            ]

            for m1, m2 in zip(dfs_mean, dfs_median):
                fig_group.add_trace(
                    go.Scatter(
                        x=x,
                        y=m1[1].values,
                        mode="lines",
                        name=f"Mean {m1[0]}",
                        line_color="#000000",
                    )
                )
                fig_group_means.add_trace(
                    go.Scatter(x=x, y=m1[1].values, mode="lines", name=f"Mean {m1[0]}")
                )
                fig_group_median.add_trace(
                    go.Scatter(x=x, y=m2[1], mode="lines", name=f"Mean {m2[0]}")
                )

        except Exception as e:
            print(e)

        s = mean.values
        s = anscombe(s)
        s = np.log(s)

        plot_line(
            np.array([s]),
            out_dir,
            label + "_" + str(df_.shape[0]),
            label + "_" + str(df_.shape[0]) + ".html",
        )
        s = CenterScaler(divide_by_std=False).transform(s)
        i = 0
        if wavelet_f0 is not None:
            CWT(
                hd=True,
                wavelet_f0=wavelet_f0,
                out_dir=out_dir,
                step_slug=label + "_" + str(df_.shape[0]) + "_" + str(i),
                animal_ids=[],
                targets=[],
                dates=[],
                n_scales=n_scales,
                vmin=0,
                vmax=3,
                enable_graph_out=True,
                sub_sample_scales=sub_sample_scales,
            ).transform([s])

        # if sfft_window is not None:
        STFT(
            enable_graph_out=True,
            sfft_window=sfft_window,
            out_dir=out_dir,
            step_slug="ANSCOMBE_" + label + "_" + str(df_.shape[0]),
            animal_ids=[],
            targets=[],
            dates=[],
        ).transform([s])

        # if dwt_w is not None:
        DWT(
            enable_graph_out=True,
            dwt_window=dwt_w,
            out_dir=out_dir,
            step_slug="ANSCOMBE_" + label + "_" + str(df_.shape[0]),
            animal_ids=[],
            targets=[],
            dates=[],
        ).transform([s])

        fig_group.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name="Mean (%d) %s" % (n, label),
                line_color="#000000",
            )
        )

        fig_group_means.add_trace(
            go.Scatter(x=x, y=mean, mode="lines", name="Mean (%d) %s" % (n, label))
        )
        fig_group_median.add_trace(
            go.Scatter(x=x, y=median, mode="lines", name="Median (%d) %s" % (n, label))
        )

        fig_group.update_layout(
            title="%d samples in category %s" % (n, label),
            xaxis_title="Time",
            yaxis_title="Activity (count)",
        )
        fig_group_means.update_layout(
            title="Mean of samples for each category",
            xaxis_title="Time",
            yaxis_title="Activity (count)",
        )
        fig_group_median.update_layout(
            title="Median of samples for each category",
            xaxis_title="Time",
            yaxis_title="Activity (count)",
        )
        traces.append(fig_group)
        # fig_group.show()

    traces.append(fig_group_means)
    traces.append(fig_group_median)
    traces = traces[::-1]  # put the median grapth first
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    # stack cwt figures
    files = list((out_dir / "_cwt").glob("*.png"))
    concatenate_images(files, out_dir)


def plot_high_dimension_db(
    out_dir, X, y, train_index, meta, clf, steps, ifold, export_fig_as_pdf
):
    """
    Plot high-dimensional decision boundary
    """
    print(f"plot_high_dimension_db {ifold}")
    try:
        db = DBPlot(clf)
        db.fit(X, y, training_indices=train_index)
        fig, ax = plt.subplots(figsize=(10.20, 12.00))
        db.plot(
            ax, generate_testpoints=True, meta=meta
        )  # set generate_testpoints=False to speed up plotting
        models_visu_dir = (
            out_dir
            / "models_visu_pca"
            / f"{type(clf).__name__}_{clf.kernel}_{steps}"
        )
        models_visu_dir.mkdir(parents=True, exist_ok=True)
        filepath = models_visu_dir / f"{ifold}.png"
        print(filepath)
        fig.savefig(filepath)
        plot_learning_curves(clf, X, y, ifold, models_visu_dir)
        db = DBPlot(clf, dimensionality_reduction=PLSRegression(n_components=2))
        db.fit(X, y, training_indices=train_index)
        fig, ax = plt.subplots(figsize=(10.20, 10.20))
        _, l1, l2 = db.plot(
            ax, generate_testpoints=True, meta=meta
        )  # set generate_testpoints=False to speed up plotting
        models_visu_dir = (
            out_dir
            / "models_visu_pls"
            / f"{type(clf).__name__}_{clf.kernel}_{steps}"
        )
        models_visu_dir.mkdir(parents=True, exist_ok=True)
        filepath = models_visu_dir / f"{ifold}.png"
        print(filepath)
        # fig.tight_layout()
        fig.savefig(
            filepath,
            bbox_extra_artists=(
                l1,
                l2,
            ),
            bbox_inches="tight",
        )
        if export_fig_as_pdf:
            filepath = models_visu_dir / f"{ifold}.pdf"
            print(filepath)
            fig.savefig(
                filepath,
                bbox_extra_artists=(
                    l1,
                    l2,
                ),
                bbox_inches="tight",
            )
        # plot_learning_curves(clf, X, y, ifold, models_visu_dir)
    except Exception as e:
        print(e)


def plot_learning_curves(clf, X, y, ifold, models_visu_dir):
    # plot learning curves for comparison
    fig, ax = plt.subplots()
    N = 10
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5)
    ax.errorbar(
        train_sizes,
        np.mean(train_scores, axis=1),
        np.std(train_scores, axis=1) / np.sqrt(N),
    )
    ax.errorbar(
        train_sizes,
        np.mean(test_scores, axis=1),
        np.std(test_scores, axis=1) / np.sqrt(N),
        c="r",
    )

    ax.legend(["Accuracies on training set", "Accuracies on test set"])
    ax.set_xlabel("Number of data points")
    ax.set_title(str(clf))
    models_visu_dir.mkdir(parents=True, exist_ok=True)
    filepath = (
        models_visu_dir
        / f"learning_curve_{ifold}_{type(clf).__name__}_{clf.kernel}.png"
    )
    print(filepath)
    plt.savefig(filepath)


def plot_fold_details(
    fold_results, meta, meta_columns, out_dir, filename="fold_details"
):
    # print(fold_results)
    # create one histogram per test fold (for loo)
    try:
        hist_list = []
        names = []
        fold_results = sorted(fold_results, key=itemgetter("target"))
        for item in fold_results:
            if len(np.array(item["y_pred_proba_test"]).shape) > 1:
                probs = [x[1] for x in item["y_pred_proba_test"]]
            else:
                probs = [x for x in item["y_pred_proba_test"]]
            m = item['meta_test'][0]
            test_fold_name = f"{int(float(m[1]))}_{int(float(m[2]))}" #TODO clean up
            names.append(test_fold_name)
            plt.clf()
            if len(np.array(item["y_pred_proba_test"]).shape) > 1:
                label = f"prob of sample (mean={np.mean([x[1] for x in item['y_pred_proba_test']]):.2f})"
            else:
                label = f"prob of sample (mean={np.mean([x for x in item['y_pred_proba_test']]):.2f})"
            h, _, _ = plt.hist(
                probs,
                density=True,
                bins=50,
                alpha=0.5,
                label=label,
            )
            hist_list.append(h)
            plt.ylabel("Density")
            plt.xlabel(
                "Probability of being unhealthy(target=1) per sample(perm of peaks)"
            )
            plt.xlim(xmin=0, xmax=1)
            plt.title(
                f"Histograms of prediction probabilities\n{test_fold_name} testing_shape={item['testing_shape']} target={item['meta_test'][0][0]}"
            )
            plt.axvline(x=0.5, color="gray", ls="--")
            plt.legend(loc="upper right")
            # plt.show()
            filename = f"histogram_of_prob_{test_fold_name}"
            out = out_dir / "loo_histograms"
            out.mkdir(parents=True, exist_ok=True)
            filepath = out / f"{filename}.png"
            print(filepath)
            plt.savefig(str(filepath))

        hist_list = np.array(hist_list)
        hist_list = hist_list + 1
        hist_list = np.log(hist_list)
        fig, ax = plt.subplots(figsize=(8.20, 7.20))
        im = ax.imshow(hist_list)
        x_axis = [
            f"{x:.1f}" for x in np.linspace(start=0, stop=1, num=hist_list.shape[1])
        ]
        ax.set_xticks(np.arange(len(x_axis)), labels=x_axis)
        ax.set_yticks(np.arange(len(names)), labels=names)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xlabel("Model Prediction Value (log)")
        # ax.set_xscale('log')

        ax.set_title("Histograms of test prediction probabilities")

        filename = f"heatmap_histogram_of_prob"
        out = out_dir / "loo_histograms"
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{filename}.png"

        fig.tight_layout()
        print(filepath)
        fig.savefig(filepath)

    except Exception as e:
        print(e)

    meta_dict = {}
    for m in meta:
        id = 0
        if "id" in meta_columns:
            id = m[meta_columns.index("id")]
        target = 0
        if "target" in meta_columns:
            target = str(m[meta_columns.index("target")])
        name = 0
        if "name" in meta_columns:
            name = m[meta_columns.index("name")]
        else:
            name = int(m[1])
        meta_dict[id] = f"{target} {name}"
    data = []
    for f in fold_results:
        i_fold = f["i_fold"]
        accuracy_train = f["accuracy_train"]
        accuracy = f["accuracy"]

        ids_test = np.unique(f["ids_test"]).astype(float)
        ids_train = np.unique(f["ids_train"]).astype(float)

        ids_test_ = np.vectorize(meta_dict.get)(ids_test)
        ids_train_ = np.vectorize(meta_dict.get)(ids_train)

        data.append([accuracy_train, accuracy, ids_test_, ids_train_])

    df = pd.DataFrame(
        data, columns=["accuracy_train", "accuracy_test", "ids_test", "ids_train"]
    )
    mean_acc_train = np.mean(df["accuracy_train"].values)
    mean_acc = np.mean(df["accuracy_test"].values)
    df = df.sort_values(by="accuracy_test")
    filepath = out_dir / f"{filename}.csv"
    df.to_csv(filepath, index=False)

    df_test = df[["accuracy_test", "accuracy_train", "ids_test"]]
    df_test = df_test.sort_values(by="accuracy_test")
    df_test.index = df["ids_test"]
    w = 0.8 * len(fold_results)
    if w > 65:
        w = 65
    ax = df_test.plot.bar(
        rot=90,
        log=False,
        figsize=(w, 7.20),
        title=f"Classifier predictions per fold n={len(fold_results)} mean_acc_train={mean_acc_train:.2f} mean_acc_test={mean_acc:.2f}",
    )
    ax.axhline(y=0.5, color="r", linestyle="--")
    try:
        for item in ax.get_xticklabels():
            if int(item.get_text().split(" ")[0].replace("[", "")) == 0:
                item.set_color("tab:blue")
    except ValueError as e:
        print(e)

    ax.set_xlabel("Fold metadata")
    ax.set_ylabel("Accuracy")
    fig = ax.get_figure()
    filepath = out_dir / f"{filename}_test.png"
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)


def build_individual_animal_pred(
    output_dir, steps, label_unhealthy, scores, ids, meta_columns, tt="test"
):
    print("build_individual_animal_pred...")
    for k, v in scores.items():
        # prepare data holder
        data_c_, data_c, data_i, data_c_prob, data_i_prob, data_m = (
            {},
            {},
            {},
            {},
            {},
            {},
        )

        for id in ids:
            data_c[id] = 0
            data_i[id] = 0
            data_c_[id] = []
            data_c_prob[id] = []
            data_i_prob[id] = []
            d = {}
            for m in meta_columns:
                d[m] = []
            data_m[id] = d

        score = scores[k]
        (
            data_dates,
            data_corr,
            data_incorr,
            prob_corr,
            prob_incorr,
            data_ids,
            data_meta,
        ) = ([], [], [], [], [], [], [])
        for s in score:
            dates = pd.to_datetime(s[f"sample_dates_{tt}"]).tolist()
            correct_predictions = s[f"correct_predictions_{tt}"]
            incorrect_predictions = s[f"incorrect_predictions_{tt}"]
            y_pred_proba_1 = np.array(s[f"y_pred_proba_{tt}"])
            y_pred_proba_0 = np.array(s[f"y_pred_proba_{tt}"])
            ids_test = s[f"ids_{tt}"]
            meta_test = s[f"meta_{tt}"]

            data_dates.extend(dates)
            data_corr.extend(correct_predictions)
            data_incorr.extend(incorrect_predictions)
            prob_corr.extend(y_pred_proba_0)
            prob_incorr.extend(y_pred_proba_1)
            data_ids.extend(ids_test)
            data_meta.extend(meta_test)

            for i in range(len(ids_test)):

                for j, m in enumerate(meta_columns):
                    data_m[ids_test[i]][m].append(meta_test[i][j])

                data_c[ids_test[i]] += correct_predictions[i]
                data_i[ids_test[i]] += incorrect_predictions[i]
                data_c_[ids_test[i]].append(correct_predictions[i])
                data_c_prob[ids_test[i]].append(y_pred_proba_0[i])
                data_i_prob[ids_test[i]].append(y_pred_proba_1[i])

        labels = list(data_c.keys())

        correct_pred = list(data_c.values())
        incorrect_pred = list(data_i.values())
        correct_pred_prob = list(data_c_prob.values())
        incorrect_pred_prob = list(data_i_prob.values())
        meta_pred = list(data_m.values())
        # make table
        df_table = pd.DataFrame(meta_pred, index=labels)
        for m in meta_columns:
            df_table[m] = [str(dict(Counter(x))) for x in df_table[m]]

        df_table["individual id"] = labels
        df_table["correct prediction"] = correct_pred
        df_table["incorrect prediction"] = incorrect_pred
        df_table["correct prediction_prob"] = correct_pred_prob
        df_table["incorrect prediction_prob"] = incorrect_pred_prob
        df_table["ratio of correct prediction (percent)"] = (
            df_table["correct prediction"]
            / (df_table["correct prediction"] + df_table["incorrect prediction"])
        ) * 100
        filename = f"table_data_{tt}_{k}.csv"
        filepath = output_dir / filename
        print(filepath)
        df_table = df_table.sort_values(
            "ratio of correct prediction (percent)", ascending=False
        )
        df_table.to_csv(filepath, index=False)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df_table.columns),
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=df_table.transpose().values.tolist(),
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )

        filename = f"table_data_{tt}_{k}.html"
        filepath = output_dir / filename
        print(filepath)
        fig.write_html(str(filepath))

        # print box plot
        plt.clf()
        max_v = max([len(x) for x in correct_pred_prob])
        correct_pred_prob_fix = []
        for item in correct_pred_prob:
            item += [np.nan] * (max_v - len(item))
            correct_pred_prob_fix.append(item)

        df = pd.DataFrame(
            {"correct_prediction_prob": correct_pred_prob_fix}, index=labels
        ).T
        df_c_p = df.apply(lambda x: x.explode() if x.name in df.columns else x)
        df_c_p = df_c_p.apply(lambda x: x.explode() if x.name in df.columns else x)
        df_c_p = df_c_p.reset_index(drop=True)
        # df_ = pd.concat([df_c_p, df_i_p], axis=1)
        df_ = df_c_p
        # df_ = df_.reindex(natsorted(df_.columns), axis=1)
        df_ = df_.astype(float)
        fig_ = plt.figure()
        boxplot = df_.boxplot(column=list(df_.columns), rot=90, figsize=(19.20, 10.80))
        boxplot.set_ylim(ymin=0, ymax=1)
        boxplot.axhline(y=0.5, color="gray", linestyle="--")
        boxplot.set_title(
            f"Classifier predictions probability ({tt}) \n per individual label_unhealthy={label_unhealthy}"
        )
        boxplot.set_xlabel("Individual")
        boxplot.set_ylabel("Probability")
        vals, names, xs = [], [], []
        for i, col in enumerate(df_.columns):
            vals.append(df_[col].values)
            names.append(col)
            xs.append(np.random.normal(i + 1, 0.04, df_[col].values.shape[0]))
        for n, (x, val) in enumerate(zip(xs, vals)):
            scatter = boxplot.scatter(
                x,
                val,
                alpha=1,
                marker="o",
                s=15,
                facecolors="none",
                edgecolors=[
                    "tab:blue" if x == 1 else "tab:red"
                    for x in list(data_c_.values())[n]
                ],
            )

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Correct prediction",
                markeredgecolor="tab:blue",
                markerfacecolor="none",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Incorrect prediction",
                markeredgecolor="tab:red",
                markerfacecolor="none",
                markersize=5,
            ),
        ]
        boxplot.legend(handles=legend_elements, loc="lower right")

        filepath = output_dir / f"predictions_per_individual_box_{k}_{steps}_{tt}.png"
        print(filepath)
        fig_.tight_layout()
        fig_.savefig(filepath)

        # print figure
        plt.clf()
        df = pd.DataFrame(
            {
                "correct prediction": correct_pred,
                "incorrect prediction": incorrect_pred,
            },
            index=labels,
        )
        df = df.sort_index()
        df = df.astype(np.double)
        df = df.sort_values("correct prediction")
        df["correct prediction"] = (
            df_table["correct prediction"]
            / (df_table["correct prediction"] + df_table["incorrect prediction"])
        ) * 100
        df["incorrect prediction"] = (
            df_table["incorrect prediction"]
            / (df_table["correct prediction"] + df_table["incorrect prediction"])
        ) * 100
        ax = df.plot.bar(
            rot=90,
            log=False,
            title=f"Classifier predictions ({tt}) per individual label_unhealthy={label_unhealthy}",
        )
        ax.set_xlabel("Animals")
        ax.set_ylabel("Number of predictions")
        fig = ax.get_figure()
        filepath = output_dir / f"predictions_per_individual_{k}_{steps}_{tt}.png"

        print(filepath)
        fig.tight_layout()
        fig.savefig(filepath)

        # figure with time
        plt.clf()
        df = pd.DataFrame(
            {
                "data_dates": data_dates,
                "data_corr": data_corr,
                "data_ids": data_ids,
                "prob_corr": prob_corr,
            }
        )
        df = df.sort_values(by="data_dates")
        dfs = [group for _, group in df.groupby(df["data_dates"].dt.strftime("%B/%Y"))]
        dfs = sorted(dfs, key=lambda x: x["data_dates"].max(axis=0))
        fig, axs = plt.subplots(
            3, int(np.ceil(len(dfs) / 3)), facecolor="white", figsize=(28.0, 10.80)
        )
        fig.suptitle(
            f"Classifier predictions ({tt}) per individual across study time label_unhealthy={label_unhealthy}",
            fontsize=14,
        )
        axs = axs.ravel()

        fig_, axs_ = plt.subplots(
            3, int(np.ceil(len(dfs) / 3)), facecolor="white", figsize=(28.0, 10.80)
        )
        fig_.suptitle(
            f"Classifier predictions probability({tt}) per individual across study time label_unhealthy={label_unhealthy}",
            fontsize=14,
        )
        axs_ = axs_.ravel()

        for ax in axs:
            ax.set_axis_off()
        for ax in axs_:
            ax.set_axis_off()

        for i, d in enumerate(dfs):
            data_c, data_u, data_c_proba, data_u_proba = {}, {}, {}, {}
            for id in ids:
                data_c[id] = 0
                data_u[id] = 0
                data_c_proba[id] = []
                data_u_proba[id] = []
            for index, row in d.iterrows():
                data_c_proba[row["data_ids"]].append(row["prob_corr"])
                if row["data_corr"] == 1:
                    data_c[row["data_ids"]] += 1
                else:
                    data_u[row["data_ids"]] += 1
            labels = list(data_c.keys())
            correct_pred = list(data_c.values())
            incorrect_pred = list(data_u.values())
            correct_pred_prob = list(data_c_proba.values())
            df = pd.DataFrame(
                {
                    "correct prediction": correct_pred,
                    "incorrect prediction": incorrect_pred,
                },
                index=labels,
            )
            df.plot.bar(
                ax=axs[i],
                rot=90,
                log=False,
                title=pd.to_datetime(d["data_dates"].values[0]).strftime("%B %Y"),
            )
            axs[i].set_ylabel("Number of predictions")
            axs[i].set_xlabel("Individual")
            axs[i].set_axis_on()
            ######################
            max_v = max([len(x) for x in correct_pred_prob])
            correct_pred_prob_fix = []
            for item in correct_pred_prob:
                item += [np.nan] * (max_v - len(item))
                correct_pred_prob_fix.append(item)

            df_c = pd.DataFrame(
                {"correct_prediction_prob": correct_pred_prob_fix}, index=labels
            ).T

            df_c_p = df_c.apply(lambda x: x.explode() if x.name in df_c.columns else x)
            df_c_p = df_c_p.apply(
                lambda x: x.explode() if x.name in df_c.columns else x
            )

            # df_c_p = df_c.explode(list(df_c.columns))
            df_c_p = df_c_p.reset_index(drop=True)
            df_ = df_c_p
            # df_ = df_.reindex(natsorted(df_.columns), axis=1)
            df_ = df_.astype(float)
            boxplot = df_.boxplot(
                column=list(df_.columns), ax=axs_[i], rot=90, figsize=(12.80, 7.20)
            )
            axs_[i].set_title(
                pd.to_datetime(d["data_dates"].values[0]).strftime("%B %Y")
            ),
            axs_[i].axhline(y=0.5, color="gray", linestyle="--")
            axs_[i].set_ylim(ymin=0, ymax=1)
            axs_[i].set_xlabel("Individual")
            axs_[i].set_ylabel("Probability of predictions")
            axs_[i].set_axis_on()
            vals, names, xs = [], [], []
            for i, col in enumerate(df_.columns):
                vals.append(df_[col].values)
                names.append(col)
                xs.append(np.random.normal(i + 1, 0.04, df_[col].values.shape[0]))
            for n, (x, val) in enumerate(zip(xs, vals)):
                scatter = boxplot.scatter(
                    x,
                    val,
                    alpha=0.9,
                    marker="o",
                    s=20,
                    facecolors="none",
                    edgecolors=[
                        "tab:blue" if x == 1 else "tab:red"
                        for x in list(data_c_.values())[n]
                    ],
                )

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Correct prediction",
                    markeredgecolor="tab:blue",
                    markerfacecolor="none",
                    markersize=5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Incorrect prediction",
                    markeredgecolor="tab:red",
                    markerfacecolor="none",
                    markersize=5,
                ),
            ]
            boxplot.legend(handles=legend_elements, loc="lower right")

        filepath = (
            output_dir
            / f"predictions_per_individual_across_study_time_{k}_{steps}_{tt}.png"
        )
        print(filepath)
        fig.tight_layout()
        fig.savefig(filepath)

        filepath = (
            output_dir
            / f"predictions_per_individual_across_study_time_box_{k}_{steps}_{tt}.png"
        )
        print(filepath)
        fig_.tight_layout()
        fig_.savefig(filepath)


def build_proba_hist(output_dir, steps, label_unhealthy, scores):
    for k in scores.keys():
        score = scores[k]
        hist_data = {}
        for label in score.keys():
            data_list = score[label]
            if len(data_list) == 0:
                continue
            label_data = []
            for elem in data_list:
                data_array = elem["test_y_pred_proba_1"]
                label_data.append(data_array)
            hist_data[label] = np.concatenate(label_data)

        plt.clf()
        plt.figure(figsize=(19.20, 10.80))
        plt.xlabel(f"Probability to be unhealthy({label_unhealthy})", size=14)
        plt.ylabel("Density", size=14)

        info = {}
        for key, value in hist_data.items():
            info[key] = hist_data[key].shape[0]

        for key, value in hist_data.items():
            plt.hist(value, density=True, bins=50, alpha=0.5, label=f"{key}")
            plt.xlim(xmin=0, xmax=1)
            plt.title(f"Histograms of prediction probabilities\n{info}")

        plt.axvline(x=0.5, color="gray", ls="--")
        plt.legend(loc="upper right")
        # plt.show()
        filename = f"histogram_of_prob_{k}_{steps}.png"
        out = output_dir / filename
        print(out)
        plt.savefig(str(out))

        df = pd.DataFrame(hist_data.keys())
        df["equal"] = df[0].apply(lambda x: (x[-1]) == (x[0]))
        df["sup"] = df[0].apply(lambda x: (x[-1]) > (x[0]))
        df["inf"] = df[0].apply(lambda x: (x[-1]) < (x[0]))

        e = np.sum(df["equal"])
        s = np.sum(df["sup"])
        i = np.sum(df["inf"])

        plt.clf()
        fig, axs = plt.subplots(2, 1, facecolor="white", figsize=(24.0, 10.80))
        axs_ = axs.ravel()
        for ax in axs_:
            ax.set_axis_off()
        for i, (k, v) in enumerate(hist_data.items()):
            a = axs[i]
            a.set_ylabel("Density", size=14)
            a.hist(v, density=True, bins=50, alpha=1, label=f"{k}")
            a.set_xlim(xmin=0, xmax=1)
            a.axvline(x=0.5, color="gray", ls="--")
            a.legend(loc="upper right")
            a.set_axis_on()

        filename = f"histogram_of_prob_{k}_{steps}_grid.png"
        fig.tight_layout()
        out = output_dir / filename
        print(out)
        plt.savefig(str(out))


def plot_histogram(x, farm_id, threshold_gap, title):
    try:
        if len(x) == 0:
            print("empty input in plot histogram!")
            return
        print("lenght=", len(x))
        print("max=", max(x))
        print("min=", min(x))
        x = pd.Series(x)

        # histogram on linear scale
        plt.subplot(211)
        plt.title(title)
        num_bins = int(max(list(set(x))))
        print("building histogram...")
        hist, bins, _ = plt.hist(x, bins=num_bins + 1, histtype="step")
        # histogram on log scale.
        # Use non-equal bin sizes, such that they look equal on log scale.
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.subplot(212)

        print("building log histogram...")
        plt.hist(x, bins=num_bins + 1, histtype="step")
        # plt.xscale('log')
        plt.yscale("log", nonposy="clip")
        print(
            "histogram_of_gap_duration_%s_%d.png"
            % (str(farm_id) + title, threshold_gap)
        )
        plt.savefig(
            "histogram_of_gap_duration_%s_%d.png"
            % (str(farm_id) + title, threshold_gap)
        )
        #plt.show()
        # plt.imsave()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    print()
