import os
import random
import shutil
from itertools import permutations

# from scipy.signal import find_peaks, find_peaks_cwt
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import typer

from utils import anscombe, check_if_hour_daylight, attribute_color


def plot_heatmap(out_dir, datetime_xaxis, matrix, y_axis, filename, title="title"):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=datetime_xaxis,
            y=y_axis,
            colorscale="Viridis",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        autosize=True,
        xaxis_title="Time (1 min bin)",
        yaxis_title="Cats",
    )
    output = out_dir / "heatmap"
    output.mkdir(parents=True, exist_ok=True)
    filepath = output / filename
    fig.write_html(str(filepath))
    print(filepath)


def create_activity_graph(
    datetime,
    colors,
    activity,
    folder,
    filename,
    title=None,
    sub_folder="training_sets_time_domain_graphs",
):
    fig = plt.figure(figsize=(24.20, 7.20))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    # plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    # plt.bar(range(0, len(activity)), activity)
    plt.bar(
        datetime, activity[0 : len(datetime)], color=colors, align="edge", width=0.01
    )
    plt.xlabel("time(1min bin)")
    plt.ylabel("Activity count")
    fig.suptitle(
        title,
        x=0.5,
        y=0.95,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )
    path = folder / sub_folder
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = f"{path}/{filename}"
    print(filepath)
    fig.savefig(filepath)


def create_training_sets(activity, metadata, out_dir, tag, filename):
    training_set = []
    training_set.extend(activity)
    training_set.append(metadata["label"])
    training_set.append(metadata["id"])
    training_set.append(metadata["imputed_days"])  # imputed days meta
    training_set.append(metadata["date"])
    training_set.append(metadata["health"])  # health
    training_set.append(metadata["target"])
    training_set.append(metadata["age"])
    training_set.append(metadata["name"])
    training_set.append(metadata["mobility_score"])

    path = out_dir / "training_sets" / tag
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = f"{path}/{filename}"
    training_str_flatten = (
        str(training_set).strip("[]").replace(" ", "").replace("None", "NaN")
    )
    print(
        f"set size is {len(training_set)}, {training_str_flatten[0:50]}.....{training_str_flatten[-50:]}"
    )
    with open(filename, "a") as outfile:
        outfile.write(training_str_flatten)
        outfile.write("\n")
    return filename


def get_cat_meta(output_dir, cat_id, output_fig=False):
    # print("getting health classification for cat id=%d" % cat_id)
    file = Path(os.getcwd()).parent / "metadata.csv"
    df = pd.read_csv(file, sep=",", nrows=55)

    if output_fig:
        df_heathy = df[df["Status"] == 0]
        df_unheathy = df[df["Status"] == 1]

        df_data = pd.concat([df_unheathy["Age"], df_heathy["Age"]], axis=1)
        df_data.columns = ["Age Unhealthy", "Age Healthy"]
        plt.clf()
        plt.cla()
        boxplot = df_data.boxplot(column=["Age Unhealthy", "Age Healthy"])
        fig_box = boxplot.get_figure()
        ax = fig_box.gca()
        ax.set_title(f"Mean age of healthy vs unhealthy cats")
        ax.set_ylabel("Age(years)")
        filename = "age.png"
        filepath = output_dir / filename
        print(filepath)
        fig_box.set_size_inches(4, 4)
        fig_box.tight_layout()
        fig_box.savefig(filepath)

        plt.clf()
        plt.cla()
        df_data_ = pd.concat(
            [df_unheathy["Mobility_Score"], df_heathy["Mobility_Score"]], axis=1
        )
        df_data_.columns = ["Mobility_Score Unhealthy ", "Mobility_Score Healthy"]
        boxplot_ = df_data_.boxplot(
            column=["Mobility_Score Unhealthy ", "Mobility_Score Healthy"]
        )
        fig_box_ = boxplot_.get_figure()
        ax_ = fig_box_.gca()
        ax_.set_title(f"Mean Mobility Score of healthy vs unhealthy cats")
        ax_.set_ylabel("Mobility Score")
        filename = "mob_score.png"
        filepath = output_dir / filename
        print(filepath)
        fig_box_.set_size_inches(4, 4)
        fig_box_.tight_layout()
        fig_box_.savefig(filepath)

    df = df[pd.notnull(df["DJD_ID"])]
    df["DJD_ID"] = df["DJD_ID"].astype(int)
    df["Status"] = df["Status"].astype(int)
    cat_meta = df.loc[df["DJD_ID"] == cat_id]

    return {
        "id": cat_meta["DJD_ID"].item(),
        "name": cat_meta["Cat"].item(),
        "age": cat_meta["Age"].item(),
        "imputed_days": -1,
        "target": -1,
        "health": cat_meta["Status"].item(),
        "mobility_score": cat_meta["Mobility_Score%"].item(),
        "label": cat_meta["Status"].item(),
    }


def build_n_peak_samples(n, rois, max_sample):
    print(f"rois n={n} shape={rois.shape}")
    stop = False
    permutation = list(permutations(range(len(rois)), n))
    if len(permutation) > max_sample:
        rois_double_idx = random.sample(permutation, k=max_sample)
    else:
        rois_double_idx = random.sample(permutation, len(permutation))  # shuffle
        stop = True

    n_peak_samples = []
    for idxs in rois_double_idx:
        new_samples = []
        for i in idxs:
            sample = rois[i]
            new_samples.append(sample)
        n_peak_samples.append(np.concatenate(new_samples))
    n_peak_samples = np.array(n_peak_samples)
    return n_peak_samples, stop


def find_region_of_interest(activity, w_size, thresh):
    print("find_region_of_interest...")
    rois = []
    df = pd.DataFrame(activity, columns=["count"])
    df["index"] = df.index
    df = df.sort_values(by=["count"], ascending=False)
    n_top = thresh
    # n_top = int(len(activity) * thresh / 100)
    print(f"n_top:{n_top}")
    df = df.iloc[0:n_top, :]
    for index, row in df.iterrows():
        i = row["index"]
        if row["count"] <= 0:
            print("negative count!")
            continue
        w_idx = list(range(i - w_size, i + w_size))
        if sum(n < 0 for n in w_idx) > 0 or i + w_size > len(
            activity
        ):  # make sure that the window is in bound
            continue
        rois.append(activity[w_idx])
    rois = np.array(rois)
    return rois


def main(data_dir, out_dir, bin, w_size, thresh, n_peaks, out_heatmap, max_sample):
    dataset_path = out_dir / "training_testing_sets"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)  # purge dataset if already created

    print(data_dir)
    print(out_dir)
    files = sorted(data_dir.glob("*.csv"))
    n_files = len(files)

    datetime_list, datetime_list_w = [], []
    activity_list, activity_list_w = [], []
    individual_list, individual_list_w = [], []
    cpt, total = 0, 0
    for i, file in enumerate(files):
        print("progress(%d/%d)..." % (i, n_files))
        print(file)
        df = pd.read_csv(file, sep=",", skiprows=range(0, 23), header=None)
        try:
            df.columns = [
                "epoch",
                "day",
                "elapsed_seconds",
                "date",
                "time",
                "activity_counts",
                "steps",
                "event_marker",
            ]
            format = "%d-%b-%y %H:%M:%S"
        except ValueError as e:  # some of the raw data is sampled at the millisecond resolution
            print(e)
            df.columns = [
                "epoch",
                "day",
                "elapsed_seconds",
                "date",
                "time",
                "activity_counts",
            ]
            format = "%d-%b-%Y %H:%M:%S.%f"

        df["date_time"] = df["date"].map(str) + " " + df["time"]
        columns_titles = [
            "epoch",
            "day",
            "elapsed_seconds",
            "activity_counts",
            "date_time",
        ]
        df = df.reindex(columns=columns_titles)
        df["date_time"] = pd.to_datetime(df["date_time"], format=format)
        df.sort_values(by="date_time")
        df["hour"] = df["date_time"].dt.hour
        df["weekday"] = np.where(df["date_time"].dt.dayofweek < 5, True, False)
        df["day_light"] = df["hour"].apply(check_if_hour_daylight)
        df = df.resample(bin, on="date_time").sum()
        df = df.reset_index()

        if bin == "T":
            df = df.iloc[: 1440 * 12, :]  # clip data to study duration 12 days
        if bin == "S":
            df = df.iloc[: 86400 * 12, :]

        df = df.set_index("date_time")
        df.reset_index(level=0, inplace=True)
        df["color"] = df.apply(attribute_color, axis=1)
        activity = df["activity_counts"].values

        rois = []
        if w_size is not None:
            rois = find_region_of_interest(activity, w_size, thresh)
            rois, stop = build_n_peak_samples(n_peaks, rois, max_sample)

        cat_id = int(file.stem.split("_")[0])
        cat_meta = get_cat_meta(out_dir, cat_id)
        df_csv = pd.DataFrame()
        df_csv["timestamp"] = df.date_time.values.astype(np.int64) // 10**9
        df_csv["date_str"] = df["date_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_csv["first_sensor_value"] = activity
        cat_meta["date"] = df["date_time"].dt.strftime("%d/%m/%Y").values[0]

        if bin == "T":
            create_activity_graph(
                df["date_time"].values,
                df["color"].values,
                activity,
                out_dir,
                f"{cat_id}_{i}_{cat_meta['name']}.png",
                title=f"{cat_id}_{cat_meta['name']}",
            )

        activity_list.append(activity)
        datetime_list.append(df["date_time"].values)
        individual_list.append(f"{cat_meta['name']} {cat_id}")

        for roi in rois:
            w = roi
            create_training_sets(w, cat_meta, out_dir, "samples", "samples.csv")
            activity_list_w.append(w)
            datetime_list_w.append(df["date_time"].values[0 : len(w)])
            individual_list_w.append(f"{cat_meta['name']} {cat_id} {i}")
            total += 1

    if out_heatmap and len(activity_list_w) == 0:
        print("create heatmap...")
        df_herd = pd.DataFrame(activity_list)
        datetime_xaxis = max(datetime_list, key=len)
        datetime_xaxis = pd.to_datetime(datetime_xaxis)
        plot_heatmap(
            out_dir,
            datetime_xaxis,
            df_herd.values,
            individual_list,
            "cats.html",
            title="Cats activity",
        )

        plot_heatmap(
            out_dir,
            datetime_xaxis,
            np.log(anscombe(df_herd.values)),
            individual_list,
            "cats_log_anscombe.html",
            title="Cats activity (LOG(ANSCOMBE())",
        )

    if out_heatmap and len(activity_list_w) > 0:
        df_herd_w = pd.DataFrame(activity_list_w)
        datetime_xaxis_w = max(datetime_list_w, key=len)
        datetime_xaxis_w = pd.to_datetime(datetime_xaxis_w)
        plot_heatmap(
            out_dir,
            datetime_xaxis_w,
            df_herd_w.values,
            individual_list_w,
            "samples.html",
            title=f"Cats activity samples total samples={total}",
        )

        plot_heatmap(
            out_dir,
            datetime_xaxis_w,
            np.log(anscombe(df_herd_w.values)),
            individual_list_w,
            "samples_log_anscombe.html",
            title=f"Cats activity samples total samples={total} (LOG(ANSCOMBE())",
        )
        del df_herd_w

    del activity_list_w
    del activity_list
    del datetime_list
    del individual_list


def run(
    data_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    bin: Literal["S", "T"] = "S",
    w_size: List[int] = [10, 30, 60, 90],
    threshs: List[int] = [10, 100, 1000],
    n_peaks: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    out_heatmap: bool = False,
    max_sample: int = 10000,
    n_job=6,
):
    """Script which builds dataset ready for ml
    Args:\n
        data_dir: Activity data directory.\n
        out_dir: Output directory.\n
        bin: Activity bin size (activity count will be summed), 'T' for minutes and 'S' for seconds .\n
        w_size: Sample lengh (if bin is S, 60 give 60 seconds sample length).\n
        thresh: Top n highest values.\n
        n_peaks: Number of peaks in dataset.\n
        out_heatmap: Enables output of visualisation heatmaps.\n
        max_sample: Maximum number of samples per cats when using n_peaks > 1.\n
        n_job: Number of threads to use.
    """
    pool = Pool(processes=n_job)

    for w in w_size:
        for n_peak in n_peaks:
            for t in threshs:
                dirname = f"{max_sample}_{t}_{str(w).zfill(3)}_{str(n_peak).zfill(3)}"
                out_dir = out_dir / dirname / "dataset"
                out_dir.mkdir(parents=True, exist_ok=True)
                pool.apply_async(
                    main,
                    (data_dir, out_dir, bin, w, t, n_peaks, out_heatmap, max_sample),
                ),

    pool.close()
    pool.join()


if __name__ == "__main__":
    typer.run(run)
    # local_run_now()
    # local_run()
    # run(
    #     data_dir=Path("/mnt/storage/scratch/axel/cats"),
    #     out_dir=Path("/mnt/storage/scratch/axel/cats/output"),
    #     n_job=28,
    # )
