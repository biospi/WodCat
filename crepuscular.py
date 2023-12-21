from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build_dataset import get_cat_data
from utils.utils import time_of_day

if __name__ == "__main__":
    data_dir = Path("E:/Cats")
    out = Path("E:/Cats")
    filename = "crepuscular"
    cat_data = get_cat_data(data_dir, "S")

    cats_time_group = []
    for i, data in enumerate(cat_data):
        print(f"{i}/{len(cat_data)}...")
        df = data[1]
        cat_id = data[0]
        df['hour'] = df.index.hour
        df["cat_id"] = cat_id
        df['time_of_day'] = df['hour'].apply(time_of_day)
        cats_time_group.append(df)

    df_all = pd.concat(cats_time_group)
    dfs = [group for _, group in df_all.groupby(["time_of_day"])]

    cat_activity_pertime = []
    colors = ['#FFD700', '#FFA07A', '#98FB98', '#FFB6C1', '#ADD8E6']
    unique_times = ['Early Morning', 'Morning', 'Noon', 'Eve', 'Night/Late Night']
    y_label = "Activity count"
    fig, axs = plt.subplots(1, len(unique_times), figsize=(10, 4), sharey=True)
    interval = 1000
    for n, df_ in enumerate(dfs):
        time_of_day = df_["time_of_day"].values[0]
        cat_activity = []
        for cat_id in np.unique(df_["cat_id"]):
            activity = df_[df_["cat_id"] == cat_id]["activity_counts"].values
            timestamp = df_[df_["cat_id"] == cat_id].index.values
            cat_activity.append(activity)

        cat_activity = pd.DataFrame(cat_activity)
        #cat_activity_pertime.append(cat_activity)

        mean_curve = cat_activity.mean(axis=0)
        lower_bound = cat_activity.quantile(0.025, axis=0)
        upper_bound = cat_activity.quantile(0.975, axis=0)
        ax = axs[n]
        ax.plot(timestamp, mean_curve, label='Mean activity', color='black')
        ax.fill_between(cat_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=colors[n],
                        label='Spread (95th percentile)')
        ax.set_xlabel('Time in seconds')
        ax.set_ylabel(y_label)
        ax.set_title(f'{time_of_day}')
        if n == 0:
            ax.legend()
        ax.grid(True)
        ax.set_xticks(timestamp[::interval])
        ax.tick_params(axis='x', rotation=45)

        #x_ticks = np.linspace(0, len(mean_curve) - 1, n_xtick, dtype=int)
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels([str(int(mean_curve.index[i])) for i in x_ticks])

        fig_, ax_ = plt.subplots()
        ax_.plot(timestamp, mean_curve, label='Mean peak', color='black')
        ax_.fill_between(cat_activity.columns.astype(int), lower_bound, upper_bound, alpha=0.6, color=colors[n], label='Spread(95th percentile)')
        ax_.set_xlabel('Time')
        ax_.set_ylabel(y_label)
        ax_.set_title(f'Mean activity with spread(95th percentile ) at {time_of_day}')
        ax_.legend()
        ax_.grid(True)
        ax_.set_xticks(timestamp[::interval])
        ax_.tick_params(axis='x', rotation=45)

        filepath = out / f'{filename}_{time_of_day}.png'.replace('/', '_').replace(' ',"_") # prevent / in 'Night/Late Night' to interfere
        print(filepath)
        fig_.savefig(filepath, bbox_inches='tight')

    fig.tight_layout()
    filepath = out / f'{filename}.png'
    print(filepath)
    fig.savefig(filepath, bbox_inches='tight')