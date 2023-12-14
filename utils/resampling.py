import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import signal


def resample_s(y, n, axis=0):
    return signal.resample(y, n, axis=axis)


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


if __name__ == "__main__":
    filepath = "E:/Data2/debug3/delmas/dataset4_mrnn_7day/activity_farmid_dbft_7_1min.csv"
    df_1min = pd.read_csv(filepath, sep=",", header=None, low_memory=False).head(1)
    df_1min = df_1min.iloc[:, 0:60]
    df_resample_10min = resample(df_1min, "10T")
    df_resample_0_7min = resample(df_1min, "0.70T")
    df_resample_0_2min = resample(df_1min, "0.20T")

    fig, axs = plt.subplots(3, 1, facecolor="white")
    fig.suptitle("Signal resampling with Pandas", fontsize=14)
    axs = axs.ravel()
    for i in range(df_1min.shape[0]):
        activity_0_7min = df_resample_0_7min.iloc[i, :].values
        activity_1min = df_1min.iloc[i, :].values
        activity_10min = df_resample_10min.iloc[i, :].values

        print(activity_0_7min)
        print(activity_1min)
        print(activity_10min)

        axs[0].plot(activity_1min)
        axs[0].set(xlabel="time", ylabel="activity")
        axs[0].set_title("Original resolution=1min")

        axs[1].plot(activity_0_7min)
        axs[1].set(xlabel="time", ylabel="activity")
        axs[1].set_title("Resampling resolution=0.70min")

        axs[2].plot(activity_10min)
        axs[2].set(xlabel="time", ylabel="activity")
        axs[2].set_title("Resampling resolution=10min")
    #fig.show()

    a_r = resample_s(activity_1min, int(np.ceil(len(activity_1min)/0.7)))
    a_r_ = resample_s(activity_1min, int(np.ceil(len(activity_1min)/10))+1)
    print("****")
    print(activity_1min)
    print(a_r)

    fig, axs = plt.subplots(3, 1, facecolor="white")
    fig.suptitle("Signal resampling with Scipi", fontsize=14)
    axs = axs.ravel()
    axs[0].plot(activity_1min)
    axs[0].set(xlabel="time", ylabel="activity")
    axs[0].set_title("Original resolution=1min")
    axs[1].plot(a_r)
    axs[1].set(xlabel="time", ylabel="activity")
    axs[1].set_title("Resampling resolution=0.7min")
    axs[2].plot(a_r_)
    axs[2].set(xlabel="time", ylabel="activity")
    axs[2].set_title("Resampling resolution=10min")

    #fig.show()