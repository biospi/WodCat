from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_dataset import get_cat_data
from utils._anscombe import anscombe
from utils._normalisation import QuotientNormalizer
from utils.utils import time_of_day


def build_crepuscular_dataset(df, out_dir, filename="samples.csv"):
    dfs = [group for _, group in df.groupby(["day"])]
    data = []
    for d in dfs:
        dfs_ = [g for _, g in d.groupby(["time_of_day"])]
        sample = []
        for d_ in dfs_:
            a_sum = d_["activity_counts"].sum()
            a_median = d_["activity_counts"].median()
            a_mean = d_["activity_counts"].mean()
            time_of_day = d_["time_of_day"].values[0]
            datetime = str(d_.index.values[0])
            health = d_["health"].values[0]
            cat_id = d_["cat_id"].values[0]
            sample.append(a_sum)
            sample.append(a_median)
            sample.append(a_mean)
        sample.append(cat_id)
        sample.append(datetime)
        sample.append(health)
        sample.append(health)

        if len(sample) != 19:#keep sample where all time of days are available
            continue
        data.append(sample)

        file_path = out_dir / filename
        file_path = file_path.as_posix()
        training_str_flatten = str(sample).strip("[]").replace(" ", "")

        with open(file_path, "a") as outfile:
            outfile.write(training_str_flatten)
            outfile.write("\n")
    meta_cols = ["cat_id", "datetime", "health", "target"]
    return meta_cols


if __name__ == "__main__":
    data_dir = Path("E:/Cats")
    out = Path("E:/Cats/crepuscular")
    filename = "crepuscular_sec_ansc"
    cat_data = get_cat_data(data_dir, "T")
    num_ticks = 6
    p = 0.95

    samples_dir = out / "dataset"
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_file = samples_dir / "samples.csv"
    if samples_file.exists():
        print(f"deleting {samples_file}")
        samples_file.unlink()

    cats_time_group = []
    for i, data in enumerate(cat_data):
        print(f"{i}/{len(cat_data)}...")
        df = data[1]
        cat_id = data[0]
        df['hour'] = df.index.hour
        df["cat_id"] = cat_id
        df['time_of_day'] = df['hour'].apply(time_of_day)
        cats_time_group.append(df)
        meta_names = build_crepuscular_dataset(df, samples_dir)

    pd.DataFrame(meta_names).to_csv(samples_dir / "meta_columns.csv", index=False)
    df_all_ = pd.concat(cats_time_group)

    for h in [0, 1]:
        df_all = df_all_[df_all_["health"] == h]
        dfs = [(group["time_of_day"].values[0], group) for _, group in df_all.groupby(["time_of_day"])]
        cat_activity_pertime = []
        colors = ['#FFD700', '#FFA07A', '#98FB98', '#FFB6C1', '#ADD8E6']
        unique_times = ['Early Morning', 'Morning', 'Noon', 'Eve', 'Night/Late Night']
        y_label = "Activity count (Anscombe)"
        fig, axs = plt.subplots(1, len(unique_times), figsize=(12, 4), sharey=True)
        interval = 1000
        dfs = sorted(dfs, key=lambda x: unique_times.index(x[0]))

        for n, item in enumerate(dfs):
            df_ = item[1]
            time_of_day = df_["time_of_day"].values[0]
            cat_activity = []
            timestamp = None
            for cat_id in np.unique(df_["cat_id"]):
                activity = df_[df_["cat_id"] == cat_id]["activity_counts"].values
                activity[activity < 0] = 0
                timestamp = df_[df_["cat_id"] == cat_id].index.values
                timestamp = pd.to_datetime(timestamp)
                cat_activity.append(activity)

            cat_activity = pd.DataFrame(cat_activity)

            labels = np.array(list(range(len(cat_activity))))
            cat_activity = QuotientNormalizer(
                out_dir=out, labels=labels, animal_ids=labels, output_graph=False, enable_qn_peak_filter=False,
            ).transform(cat_activity.values)

            cat_activity = pd.DataFrame(cat_activity)
            cat_activity = anscombe(cat_activity)
            #cat_activity = np.log(cat_activity)
            #cat_activity_pertime.append(cat_activity)

            mean_curve = cat_activity.mean(axis=0).values
            lower_bound = cat_activity.quantile(0.025, axis=0)
            upper_bound = cat_activity.quantile(p, axis=0)
            ax = axs[n]
            ax.plot(mean_curve, label='Mean activity', color='black')
            ax.fill_between(cat_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=colors[n],
                            label=f'Spread ({int(p*100)}th percentile)')
            ax.set_xlabel('Time in seconds')
            ax.set_ylabel(y_label)
            ax.set_title(f'{time_of_day} h={h}')
            if n == 0:
                ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
            total_points = len(timestamp)
            tick_positions = [int(i * (total_points - 1) / (num_ticks - 1)) for i in range(num_ticks)]
            ax.xaxis.set_major_locator(plt.FixedLocator(tick_positions))
            ax.set_xticklabels([timestamp[i].strftime('%H:%M') for i in tick_positions])

            fig_, ax_ = plt.subplots()
            ax_.plot(mean_curve, label='Mean activity', color='black')
            ax_.fill_between(cat_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=colors[n], label=f'Spread({int(p*100)}th percentile)')
            ax_.set_xlabel('Time')
            ax_.set_ylabel(y_label)
            ax_.set_title(f'Mean activity with spread({int(p*100)}th percentile ) at {time_of_day} health={h}')
            ax_.legend()
            ax_.grid(True)
            ax_.tick_params(axis='x', rotation=45)
            ax_.xaxis.set_major_locator(plt.FixedLocator(tick_positions))
            ax_.set_xticklabels([timestamp[i].strftime('%H:%M') for i in tick_positions])
            fig_.autofmt_xdate()
            filepath = out / f'{filename}_{time_of_day}_{h}.png'.replace('/', '_').replace(' ',"_") # prevent / in 'Night/Late Night' to interfere
            print(filepath)
            fig_.savefig(filepath, bbox_inches='tight')

        fig.autofmt_xdate()
        fig.tight_layout()
        filepath = out / f'{filename}_{h}.png'
        print(filepath)
        fig.savefig(filepath, bbox_inches='tight')
