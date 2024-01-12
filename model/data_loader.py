import datetime
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def resample_s(df_activity_window, sampling):
    resampled = []
    for i in range(df_activity_window.shape[0]):
        row = df_activity_window.iloc[i, :]
        row = signal.resample(row, sampling)
        resampled.append(row)
    df_ = pd.DataFrame(resampled)
    df_.columns = list(range(df_.shape[1]))
    return df_


def resample(df_activity_window, sampling):
    resampled = []
    for i in range(df_activity_window.shape[0]):
        row = df_activity_window.iloc[i, :]
        mins = [datetime.datetime.today() + datetime.timedelta(minutes=1 * x) for x in range(0, len(row))]
        row.index = mins
        df_activity_window_ = row.resample(sampling).sum()
        resampled.append(df_activity_window_)
    df_ = pd.DataFrame(resampled)
    df_.columns = list(range(df_.shape[1]))
    return df_


def load_activity_data(
    out_dir,
    meta_columns,
    filepath,
    class_healthy,
    class_unhealthy,
    preprocessing_steps=None,
    farm=None,
    sampling='T',
    individual_to_ignore=[],
    individual_to_keep=[],
    plot_s_distribution=False,
    sample_date_filter = None
):
    print(f"load activity from datasets...{filepath}")
    data_frame = pd.read_csv(filepath, sep=",", header=None, low_memory=False)
    data_frame = data_frame.drop_duplicates()
    data_frame = data_frame.astype(
        dtype=float, errors="ignore"
    )  # cast numeric values as float
    data_point_count = data_frame.shape[1]
    hearder = [str(n) for n in range(0, data_point_count)]

    for i, m in enumerate(meta_columns[::-1]):
        hearder[-i - 1] = m
    data_frame.columns = hearder
    # data_frame['label'] = data_frame['label'].astype(int).astype(str)
    if "name" in data_frame.columns:
        data_frame["name"] = data_frame["name"].astype(str).str.replace("'", "")
    data_frame["date"] = data_frame["date"].astype(str).str.replace("'", "")
    # cast transponder ids to string instead of float
    data_frame["id"] = data_frame["id"].astype(str).str.split(".", expand=True, n=0)[0]

    if sample_date_filter is not None:
        data_frame["datetime"] = pd.to_datetime(data_frame['date'])
        data_frame = data_frame[data_frame["datetime"] > pd.Timestamp(sample_date_filter)]
        data_frame = data_frame.drop('datetime', 1)

    if len(individual_to_ignore) > 0:
        data_frame = data_frame.loc[~data_frame['name'].isin(individual_to_ignore)]#todo fix

    if len(individual_to_keep) > 0:
        data_frame = data_frame.loc[data_frame['name'].isin(individual_to_keep)]

    data_frame = data_frame.dropna(
        subset=data_frame.columns[: -len(meta_columns)], how="all"
    )

    if len(preprocessing_steps) > 0:
        if "ZEROPAD" in preprocessing_steps[0]:
            data_frame = data_frame.fillna(0)

        if "LINEAR" in preprocessing_steps[0]:
            data_frame.iloc[:, : -len(meta_columns)] = data_frame.iloc[
                :, : -len(meta_columns)
            ].astype(np.float16).interpolate(axis=1, limit_direction="both")

    data_frame = data_frame.dropna()

    data_frame['id'] = data_frame['id'].astype(np.float16)

    # clip negative values
    data_frame[data_frame.columns.values[: -len(meta_columns)]] = data_frame[
        data_frame.columns.values[: -len(meta_columns)]
    ].astype(float).clip(lower=0)

    data_frame["target"] = data_frame["target"].astype(int)
    data_frame["label"] = data_frame["label"].astype(str)
    new_label = []
    # data_frame_health = data_frame.copy()
    for v in data_frame["label"].values:
        if v in class_healthy:
            new_label.append(0)
            continue
        if v in class_unhealthy:
            new_label.append(1)
            continue
        new_label.append(-1)

    data_frame["health"] = new_label

    data_frame_labeled = pd.get_dummies(data_frame, columns=["label"])
    flabels = [x for x in data_frame_labeled.columns if "label" in str(x)]

    print(f"flabels={flabels}")
    for i, flabel in enumerate(flabels):
        #data_frame_labeled[flabel] = data_frame_labeled[flabel] * (i + 1)
        data_frame_labeled.loc[:, flabel] = data_frame_labeled[flabel] * (i + 1)
        data_frame.loc[:, "target"] = data_frame["target"] + data_frame_labeled[flabel]

    # store all samples for later testing after binary fitting
    labels = data_frame["label"].drop_duplicates().values
    samples = {}

    for label in labels:
        df = data_frame[data_frame["label"] == label]
        samples[label] = df

    if plot_s_distribution:
        plot_samples_distribution(out_dir, samples, f"distrib_all_samples_{farm}.png")

    class_count = {}
    label_series = dict(data_frame[["target", "label"]].drop_duplicates().values)
    label_series_inverse = dict((v, k) for k, v in label_series.items())
    print(label_series_inverse)
    print(label_series)
    for k in label_series.keys():
        class_count[str(label_series[k]) + "_" + str(k)] = data_frame[
            data_frame["target"] == k
        ].shape[0]
    print(class_count)
    # drop label column stored previously, just keep target for ml
    df_meta = data_frame[meta_columns]
    print(data_frame)

    return (
        data_frame,
        df_meta,
        class_healthy,
        class_unhealthy,
        label_series,
        samples
    )


def parse_param_from_filename(file):
    split = file.split("/")[-1].split(".")[0].split("_")
    # activity_delmas_70101200027_dbft_1_1min
    sampling = ""
    days = 0
    farm_id = "farm_id"
    option = ""
    for s in split:
        if "day" in s:
            days = int(s[0])
            break

    return days, farm_id, option, sampling


def plot_samples_distribution(out_dir, samples_, filename):
    print("plot_samples_distribution...")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_data = samples_.copy()

    #print(sample_data)
    # bar plot
    d = []
    for key, value in sample_data.items():
        new_col = [f"{int(float(key))}_{x}" for x in value["id"].astype(int)]
        value.loc[:, "id"] = new_col
        d.append(dict(value["id"].value_counts()))

    c = Counter()
    for dct in d:
        c.update(dct)
    c = dict(c)

    df_list = []
    for k, v in c.items():
        split = k.split("_")
        df_list.append([split[0], split[1], v])
    df = pd.DataFrame(df_list, columns=["Labels", "id", "count"])
    df.loc[:, "id"] = df["id"].str.split(".").str[0]

    info = {}
    for l in sample_data.keys():
        total = df[df["Labels"] == l]["count"].sum()
        info[l] = total

    plt.clf()
    df.groupby(["id", "Labels"]).sum().unstack().plot(
        kind="bar",
        y="count",
        figsize=(9.20, 9.20),
        stacked=True,
        xlabel="Transponders",
        ylabel="Number of samples",
        title=f"Distribution of samples across transponders\n{str(info)}",
    )
    filepath = str(out_dir / filename)
    print(filepath)
    plt.savefig(filepath, bbox_inches="tight")