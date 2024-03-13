import os
import random
import shutil
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import typer
from plotnine import ggplot, aes, geom_jitter, stat_summary, theme, element_text
from utils._anscombe import anscombe
from utils.utils import time_of_day_


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
    activity,
    folder,
    filename,
    title=None,
    sub_folder="training_sets_time_domain_graphs",
):
    fig = plt.figure(figsize=(6, 4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    # plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    # plt.bar(range(0, len(activity)), activity)
    plt.bar(
        datetime, activity[0 : len(datetime)], align="edge", width=0.01
    )
    plt.xlabel("time(1min bin)")
    plt.ylabel("Activity count")
    plt.xticks(rotation=45)
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


def create_training_sets(run_id, activity, timestamp, metadata, max_sample, n_peak, w_size, thresh, out_dir, filename):
    meta_names = []
    training_set = []
    training_set.extend(activity)
    for n, t in enumerate(timestamp):
        training_set.append(t)
        meta_names.append(f"peak{n}_datetime")
    training_set.append(metadata["label"])
    meta_names.append("label")
    training_set.append(metadata["id"])
    meta_names.append("id")
    training_set.append(metadata["date"])
    meta_names.append("date")
    training_set.append(metadata["health"])  # health
    meta_names.append("health")
    training_set.append(metadata["target"])
    meta_names.append("target")
    training_set.append(metadata["age"])
    meta_names.append("age")
    training_set.append(metadata["name"])
    meta_names.append("name")
    training_set.append(metadata["mobility_score"])
    meta_names.append("mobility_score")
    training_set.append(max_sample)
    meta_names.append("max_sample")
    training_set.append(n_peak)
    meta_names.append("n_peak")
    training_set.append(w_size)
    meta_names.append("w_size")
    training_set.append(thresh)
    meta_names.append("n_top")
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    filepath = filepath.as_posix()
    training_str_flatten = (
        str(training_set).strip("[]").replace(" ", "").replace("None", "NaN")
    )
    # print(
    #     f"[{run_id}] sample size is {len(training_set)}: {training_str_flatten[0:50]}.....{training_str_flatten[-50:]}"
    # )
    with open(filepath, "a") as outfile:
        outfile.write(training_str_flatten)
        outfile.write("\n")
    return filepath, meta_names


def get_cat_meta(output_dir, cat_id, output_fig=True):
    # print("getting health classification for cat id=%d" % cat_id)
    file = Path(os.getcwd()) / "metadata.csv"
    df = pd.read_csv(file, sep=",", nrows=55)
    df_ = df.copy()

    if output_fig:
        for col in ["Age", "Mobility_Score"]:
            df_["Status"] = df_["Status"].replace(0, "No DJD")
            df_["Status"] = df_["Status"].replace(1, "DJD")

            g = (
                    ggplot(df_)  # defining what data to use
                    + aes(
                x="Status", y=col, color="Status"
            )  # defining what variable to use
                    + geom_jitter()  # defining the type of plot to use
                    + stat_summary(geom="crossbar", color="black", width=0.2)
                    + theme(
                subplots_adjust={"right": 0.82}, axis_text_x=element_text(angle=45, hjust=1), legend_position='none'
            )
            )

            fig = g.draw()
            ax = fig.gca()
            ax.set_title(f"Cat {col}")
            ax.set_ylabel("Mobility Score")
            if col == "Age":
                ax.set_ylabel("Age(years)")
            filename = f"{col}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
            if not filepath.exists():
                print(filepath)
                fig.set_size_inches(3, 4)
                fig.tight_layout()
                fig.savefig(filepath)

    df = df[pd.notnull(df["DJD_ID"])]
    df["DJD_ID"] = df["DJD_ID"].astype(int)
    df["Status"] = df["Status"].astype(int)
    cat_meta = df.loc[df["DJD_ID"] == cat_id]

    return {
        "id": cat_meta["DJD_ID"].item(),
        "name": cat_meta["Cat"].item(),
        "age": cat_meta["Age"].item(),
        "target": -1,
        "health": cat_meta["Status"].item(),
        "mobility_score": cat_meta["Mobility_Score%"].item(),
        "label": cat_meta["Status"].item(),
    }


def build_n_peak_samples(run_id, tot, n_peak, rois, rois_timestamp, max_sample):
    print(f"[{run_id}/{tot}] number of peaks is 1, sample shape is{rois.shape}")
    idxs_peaks = np.arange(rois.shape[0])
    permutation = list(permutations(idxs_peaks, n_peak))
    try:
        rois_idxs = random.sample(permutation, k=max_sample)
    except ValueError as e:
        print(e)
        print(f"There are less samples than max_sample={max_sample}")
        rois_idxs = permutation

    #build augmented sample by concatenating permutations of peaks
    n_peak_samples = []
    for idxs in rois_idxs:
        new_samples = []
        timestamps = []
        for i in idxs:
            sample = rois[i]
            new_samples.append(sample)
            timestamp = str(rois_timestamp[i])
            timestamps.append(timestamp)
        activity = np.concatenate(new_samples)
        s = activity.tolist() + timestamps
        n_peak_samples.append(s)
    n_peak_samples = np.array(n_peak_samples)
    print(f"[{run_id}/{tot}] number of peaks is {n_peak}, sample shape is{n_peak_samples.shape}")
    return n_peak_samples


def find_region_of_interest(run_id, tot, timestamp, activity, w_size, thresh):
    print(f"[{run_id}/{tot}] find_region_of_interest...")
    rois = []
    rois_timestamp = []
    df = pd.DataFrame(activity, columns=["count"])
    df["index"] = df.index
    df_sorted = df.sort_values(by=["count"], ascending=False)
    n_top = thresh
    # n_top = int(len(activity) * thresh / 100)
    df_sorted = df_sorted.iloc[0:n_top, :]
    for index, row in df_sorted.iterrows():
        i = row["index"]
        if row["count"] <= 0:
            print("negative count!")
            continue
        w = int(w_size/2)
        w_idx = list(range(i - w, i + w))
        if sum(n < 0 for n in w_idx) > 0 or i + w_size > len(
            activity
        ):  # make sure that the window is in bound
            continue
        roi = activity[w_idx]
        rois.append(roi)
        if timestamp is not None:
            rois_timestamp.append(timestamp[i])
    rois = np.array(rois).astype(np.int32)
    return rois, rois_timestamp


def format_raw_data(df, bin):
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
        #print(e)
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
    #df["hour"] = df["date_time"].dt.hour
    #df["weekday"] = np.where(df["date_time"].dt.dayofweek < 5, True, False)
    #df["day_light"] = df["hour"].apply(check_if_hour_daylight)
    df = df.resample(bin, on="date_time").sum()
    df = df.reset_index()

    if bin == "T":
        df = df.iloc[: 1440 * 12, :]  # clip data to study duration 12 days
    if bin == "S":
        df = df.iloc[: 86400 * 12, :]

    df = df.set_index("date_time")
    #df.reset_index(level=0, inplace=True)
    #df["color"] = df.apply(attribute_color, axis=1)
    df["epoch"] = df["epoch"].astype(np.int32)
    df["day"] = df["day"].astype(np.int8)
    df["elapsed_seconds"] = df["elapsed_seconds"].astype(np.int32)
    df["activity_counts"] = df["activity_counts"].astype(np.int32)
    df["day"] = df.index.day
    df['hour'] = df.index.hour
    df['time_of_day'] = df['hour'].apply(time_of_day_)
    return df


def main(time_of_day, cat_data, out_dir, bin, w_size, thresh, n_peak, out_heatmap, max_sample, run_id, tot):
    print(f"[{run_id}] progress[{run_id}/{tot}]...")

    datetime_list, datetime_list_w = [], []
    activity_list, activity_list_w = [], []
    individual_list, individual_list_w = [], []
    cpt, total = 0, 0
    for i, df in enumerate(cat_data):
        print(f"[{run_id}/{tot}] progress[{i}/{len(cat_data)}]...")

        if time_of_day != 'All':
            df = df[df["time_of_day"] == time_of_day]

        cat_id = df["cat_id"].values[0]
        activity = df["activity_counts"].values
        timestamp = df.index

        rois = []
        if w_size is not None:
            rois, rois_timestamp = find_region_of_interest(run_id, tot, timestamp, activity, w_size, thresh)
            rois = build_n_peak_samples(run_id, tot, n_peak, rois, rois_timestamp, max_sample)
            rois_timestamp = rois[:, -n_peak:]
            rois = rois[:, :-n_peak].astype(int)

        cat_meta = get_cat_meta(out_dir, cat_id)
        cat_meta["date"] = df.index.strftime("%d/%m/%Y").values[0]

        if bin == "T":
            create_activity_graph(
                df.index.values,
                activity,
                out_dir,
                f"{cat_id}_{i}.png",
                title=f"{cat_id}",
            )

        if out_heatmap:
            activity_list.append(activity)
            datetime_list.append(df.index.values)
            individual_list.append(f"{cat_id}")

        for roi, timestamp in zip(rois, rois_timestamp):
            _, meta_names = create_training_sets(run_id, roi, timestamp, cat_meta, max_sample, n_peak, w_size, thresh, out_dir, "samples.csv")
            if out_heatmap:
                activity_list_w.append(roi)
                datetime_list_w.append(df.index.values[0 : len(roi)])
                individual_list_w.append(f"{cat_id} {i}")
            total += 1
        pd.DataFrame(meta_names).to_csv(out_dir / "meta_columns.csv", index=False)

    if out_heatmap:
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
    return meta_names


def get_cat_data(data_dir, bin, subset=None):
    print("Loading cat data...")
    if bin not in ["S", "T"]:
        print(f"bin value must be 'S' or 'T'. {bin} is not supported!")
    files = sorted(data_dir.glob("*.csv"))
    if subset is not None:
        files = files[0:subset]

    # new = []
    # for f in files:
    #     if 11 == int(f.name.split('_')[0]) or 13 == int(f.name.split('_')[0]):
    #         new.append(f)
    #     if 36 == int(f.name.split('_')[0]) or 75 == int(f.name.split('_')[0]):
    #         new.append(f)
    # files = new

    dfs = []
    for i, file in enumerate(files):
        print(f"progress[{i}/{len(files)}]...")
        print(f"reading file: {file}")
        df = pd.read_csv(file, sep=",", skiprows=range(0, 23), header=None)
        cat_id = int(file.stem.split("_")[0])
        cat_name = file.stem.split("_")[1]
        individual_to_ignore = ["MrDudley", "Oliver_F", "Lucy"]
        if cat_name in individual_to_ignore:
            continue
        df = format_raw_data(df, bin)
        cat_meta = get_cat_meta(data_dir, cat_id)
        df["health"] = cat_meta["health"]
        df["age"] = cat_meta["age"]
        df["cat_id"] = cat_id
        dfs.append(df)
    return dfs


def run(
    data_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_path: Path = Path("dataset.csv"),
    bin: str = "S",
    w_size: List[int] = [10, 30, 60, 90],
    threshs: List[int] = [10, 100, 1000],
    n_peaks: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    day_windows: List[str] = ['All', 'Day', 'Night'],
    out_heatmap: bool = False,
    max_sample: int = 100,
    n_job: int = 2,
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
    #pool = Pool(processes=n_job)

    tot = len(w_size) * len(n_peaks) * len(threshs) * len(day_windows)
    cpt = 0

    print(f"dataset_path={dataset_path}")
    if dataset_path.exists():
        print(f"loading {dataset_path}")
        df_data = pd.read_csv(dataset_path, index_col="date_time")
        df_data.index = pd.to_datetime(df_data.index)
        cat_data = [group for _, group in df_data.groupby(["cat_id"])]
    else:
        cat_data = get_cat_data(data_dir, bin)
        dataset_path = f"dataset_{bin}.csv"
        print(f"saving {dataset_path}...")
        pd.concat(cat_data).to_csv(dataset_path, index=True)
        #print("done.")

    datasets = []
    for t in threshs:
        for w in w_size:
            for n_peak in n_peaks:
                for time_of_day in day_windows:
                    dirname = f"{max_sample}_{t}_{str(w).zfill(3)}_{str(n_peak).zfill(3)}"
                    if time_of_day is not None:
                        dirname = f"{time_of_day}_{max_sample}_{t}_{str(w).zfill(3)}_{str(n_peak).zfill(3)}"

                    out_dataset_dir = out_dir / dirname / "dataset"
                    datasets.append(out_dataset_dir / "samples.csv")
                    if out_dataset_dir.exists():
                        shutil.rmtree(out_dataset_dir)  # purge dataset if already created
                    # pool.apply_async(
                    #     main,
                    #     (cat_data, out_dataset_dir, bin, w, t, n_peak, out_heatmap, max_sample, cpt, tot),
                    # ), #TODO Fix multiprocessing doesn't work for large amount of data (dataset ids ~3GB)
                    main(time_of_day, cat_data, out_dataset_dir, bin, w, t, n_peak, out_heatmap, max_sample, cpt, tot)
                    cpt += 1
    # pool.close()
    # pool.join()
    return datasets


if __name__ == "__main__":
    typer.run(run)

