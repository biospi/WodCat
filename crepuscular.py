from pathlib import Path

import numpy as np
import pandas as pd

import run_ml
import boot_roc_curve
from build_dataset import get_cat_data, find_region_of_interest
from utils.utils import time_of_day_


def build_crepuscular_dataset(df, out_dir, filename="samples.csv", w_size=30, n_top=10):
    #dfs = [group for _, group in df.groupby(["day"])]
    data = []
    #for d in dfs:
    dfs_ = [g for _, g in df.groupby(["time_of_day"])]

    for d_ in dfs_:
        activity = d_["activity_counts"].values
        rois, _ = find_region_of_interest(None, activity, w_size, n_top)
        time_of_day = d_["time_of_day"].values[0]
        health = d_["health"].values[0]
        cat_id = d_["cat_id"].values[0]
        label = health #ml pipeline need this col need to represent the health status!
        name = cat_id #ml pipeline need this col
        target = health #ml pipeline need this col
        for r in rois:
            sample = r.tolist()
            date = df.index.strftime("%d/%m/%Y").values[0]
            sample.append(cat_id)
            sample.append(date)
            sample.append(name)
            sample.append(label)
            max_sample = -1
            sample.append(max_sample)#ml pipeline need this col
            n_peak = 1
            sample.append(n_peak)#ml pipeline need this col
            sample.append(w_size)#ml pipeline need this col
            sample.append(n_top)#ml pipeline need this col
            sample.append(time_of_day)
            sample.append(health)
            sample.append(target)
            data.append(sample)

            file_path = out_dir / filename
            file_path = file_path.as_posix()
            training_str_flatten = str(sample).strip("[]").replace(" ", "")

            with open(file_path, "a") as outfile:
                outfile.write(training_str_flatten)
                outfile.write("\n")
    meta_cols = ["id", "date", "name", "label", "max_sample", "n_peak", "w_size", "n_top", "time_of_day", "health", "target"]
    cols = np.arange(0, len(data[0])-len(meta_cols)).tolist() + meta_cols
    data = pd.DataFrame(data, columns=cols)
    return meta_cols, data


def ml(samples_dir, n_bootstrap=100, n_job=5):
    dataset = samples_dir / "samples.csv"
    meta_columns_file = samples_dir / "meta_columns.csv"
    meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
    print(f"dataset={dataset}")
    print(f"meta_columns={meta_columns}")

    out_dir = samples_dir.parent / "ml"
    print("Running machine learning pipeline...")
    for preprocessing_steps in [
        ["QN"]
    ]:
        out_ml_dir = run_ml.run(
            preprocessing_steps=preprocessing_steps,
            meta_columns=meta_columns,
            dataset_filepath=dataset,
            pre_visu=True,
            out_dir=out_dir,
            n_job=n_job,
        )

        boot_roc_curve.main(
            out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
        )


if __name__ == "__main__":
    #init
    data_dir = Path("E:/Cats")
    #Get data from raw csv
    cat_data = get_cat_data(data_dir, "S")
    num_ticks = 6
    p = 0.95
    w_size = 30
    n_top = 10
    time_of_day_list = ["Day", "Night"]

    for tod in time_of_day_list:
        out = Path(f"E:/Cats/paper_2/crepuscular_{w_size}_{n_top}_{tod}")
        #filename = "crepuscular_sec_ansc"
        samples_dir = out / "dataset"
        samples_dir.mkdir(parents=True, exist_ok=True)
        samples_file = samples_dir / "samples.csv"
        if samples_file.exists():
            print(f"deleting {samples_file}")
            samples_file.unlink()

        #Start analysis
        cats_time_group = []
        for i, data in enumerate(cat_data):
            df = data[1]
            cat_id = data[0]
            print(f"cat_id={cat_id} {i}/{len(cat_data)}...")
            df['hour'] = df.index.hour
            df["cat_id"] = cat_id
            df['time_of_day'] = df['hour'].apply(time_of_day_)
            df = df[df["time_of_day"] == tod]
            cats_time_group.append(df)
            meta_columns, _ = build_crepuscular_dataset(df, samples_dir, w_size=w_size, n_top=n_top)

        pd.DataFrame(meta_columns).to_csv(samples_dir / "meta_columns.csv", index=False)

        ml(samples_dir)

            # df_all_ = pd.concat(cats_time_group)
            # for h in [0, 1]:
            #     df_all = df_all_[df_all_["health"] == h]
            #     dfs = [(group["time_of_day"].values[0], group) for _, group in df_all.groupby(["time_of_day"])]
            #     cat_activity_pertime = []
            #     colors = ['#FFD700', '#FFA07A', '#98FB98', '#FFB6C1', '#ADD8E6']
            #     unique_times = ['Early Morning', 'Morning', 'Afternoon', 'Night']
            #     y_label = "Activity count (Anscombe)"
            #     fig, axs = plt.subplots(1, len(unique_times), figsize=(12, 4), sharey=True)
            #     interval = 1000
            #     dfs = sorted(dfs, key=lambda x: unique_times.index(x[0]))
            #
            #     for n, item in enumerate(dfs):
            #         df_ = item[1]
            #         time_of_day = df_["time_of_day"].values[0]
            #         cat_activity = []
            #         timestamp = None
            #         for cat_id in np.unique(df_["cat_id"]):
            #             activity = df_[df_["cat_id"] == cat_id]["activity_counts"].values
            #             activity[activity < 0] = 0
            #             timestamp = df_[df_["cat_id"] == cat_id].index.values
            #             timestamp = pd.to_datetime(timestamp)
            #             cat_activity.append(activity)
            #
            #         cat_activity = pd.DataFrame(cat_activity)
            #
            #         labels = np.array(list(range(len(cat_activity))))
            #         cat_activity = QuotientNormalizer(
            #             out_dir=out, labels=labels, animal_ids=labels, output_graph=False, enable_qn_peak_filter=False,
            #         ).transform(cat_activity.values)
            #
            #         cat_activity = pd.DataFrame(cat_activity)
            #         cat_activity = anscombe(cat_activity)
            #         #cat_activity = np.log(cat_activity)
            #         #cat_activity_pertime.append(cat_activity)
            #
            #         mean_curve = cat_activity.mean(axis=0).values
            #         lower_bound = cat_activity.quantile(0.025, axis=0)
            #         upper_bound = cat_activity.quantile(p, axis=0)
            #         ax = axs[n]
            #         ax.plot(mean_curve, label='Mean activity', color='black')
            #         ax.fill_between(cat_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=colors[n],
            #                         label=f'Spread ({int(p*100)}th percentile)')
            #         ax.set_xlabel('Time in seconds')
            #         ax.set_ylabel(y_label)
            #         ax.set_title(f'{time_of_day} h={h}')
            #         if n == 0:
            #             ax.legend()
            #         ax.grid(True)
            #         ax.tick_params(axis='x', rotation=45)
            #         total_points = len(timestamp)
            #         tick_positions = [int(i * (total_points - 1) / (num_ticks - 1)) for i in range(num_ticks)]
            #         ax.xaxis.set_major_locator(plt.FixedLocator(tick_positions))
            #         ax.set_xticklabels([timestamp[i].strftime('%H:%M') for i in tick_positions])
            #
            #         fig_, ax_ = plt.subplots()
            #         ax_.plot(mean_curve, label='Mean activity', color='black')
            #         ax_.fill_between(cat_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=colors[n], label=f'Spread({int(p*100)}th percentile)')
            #         ax_.set_xlabel('Time')
            #         ax_.set_ylabel(y_label)
            #         ax_.set_title(f'Mean activity with spread({int(p*100)}th percentile ) at {time_of_day} health={h}')
            #         ax_.legend()
            #         ax_.grid(True)
            #         ax_.tick_params(axis='x', rotation=45)
            #         ax_.xaxis.set_major_locator(plt.FixedLocator(tick_positions))
            #         ax_.set_xticklabels([timestamp[i].strftime('%H:%M') for i in tick_positions])
            #         fig_.autofmt_xdate()
            #         filepath = out / f'{filename}_{time_of_day}_{h}.png'
            #         print(filepath)
            #         fig_.savefig(filepath, bbox_inches='tight')
            #
            #     fig.autofmt_xdate()
            #     fig.tight_layout()
            #     filepath = out / f'{filename}_{h}.png'
            #     print(filepath)
            #     fig.savefig(filepath, bbox_inches='tight')
