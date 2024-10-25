import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from sys import exit

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycwt as wavelet
import pywt
from plotly.subplots import make_subplots
from scipy import signal
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
# from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (
    recall_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from utils.utils import concatenate_images

matlab = None


def plot_cwt_power_sidebyside(
    filename_sub,
    step_slug,
    output_samples,
    class_healthy_label,
    class_unhealthy_label,
    idx_healthy,
    idx_unhealthy,
    coi_line_array,
    df_timedomain,
    graph_outputdir,
    power_cwt_healthy,
    power_cwt_unhealthy,
    scales,
    ntraces=3,
    title="cwt_power_by_target",
    stepid=10,
    meta_size=None,
    format_xaxis=False,
):
    total_healthy = df_timedomain[df_timedomain["health"] == 0].shape[0]
    total_unhealthy = df_timedomain[df_timedomain["health"] == 1].shape[0]
    plt.clf()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(19.2, 7.2))


    df_healthy = df_timedomain[df_timedomain["health"] == 0].iloc[:, :-meta_size].values
    df_unhealthy = (
        df_timedomain[df_timedomain["health"] == 1].iloc[:, :-meta_size].values
    )

    ymin = min([np.nanmax(df_healthy), np.nanmin(df_unhealthy)])
    ymax = max([np.nanmax(df_healthy), np.nanmax(df_unhealthy)])

    ticks = get_time_ticks(df_healthy.shape[1])

    for row in df_healthy:
        ax1.plot(ticks, row)
        ax1.set(xlabel="", ylabel="activity")
        ax1.set_title(
            "Healthy(%s) animals %d / displaying %d"
            % (class_healthy_label, total_healthy, df_healthy.shape[0])
        )
        ax1.set_ylim([ymin, ymax])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.DayLocator())

    for row in df_unhealthy:
        ax2.plot(ticks, row)
        ax2.set(xlabel="", ylabel="activity")
        ax2.set_yticks(ax2.get_yticks().tolist())
        ax2.set_xticklabels(ticks, fontsize=12)
        ax2.set_title(
            "Unhealthy(%s) animals %d / displaying %d"
            % (class_unhealthy_label, total_unhealthy, df_unhealthy.shape[0])
        )
        ax2.set_ylim([ymin, ymax])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.DayLocator())

    imshow_y_axis = []
    p0 = power_cwt_healthy.copy()

    imshow_y_axis.append(np.nanmin(p0))
    imshow_y_axis.append(np.nanmax(p0))
    p1 = power_cwt_unhealthy.copy()

    imshow_y_axis.append(np.nanmin(p1))
    imshow_y_axis.append(np.nanmax(p1))

    vmin, vmax = min(imshow_y_axis), max(imshow_y_axis)

    pos0 = ax3.imshow(
        p0,
        extent=[0, p0.shape[1], p0.shape[0], 1],
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(pos0, ax=ax3)

    pos1 = ax4.imshow(
        p1,
        extent=[0, p1.shape[1], p1.shape[0], 1],
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(pos1, ax=ax4)

    ax3.set_aspect("auto")
    ax3.set_title(
        "Healthy(%s) animals elem wise average of %d cwts (ANSCOMBE)"
        % (class_healthy_label, df_healthy.shape[0])
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel(f"Wave length of wavelet (in minute) (n_scales={len(scales)})")
    # ax3.set_yscale('log')

    ax4.set_aspect("auto")
    ax4.set_title(
        "Unhealthy(%s) animals elem wise average of %d cwts (ANSCOMBE)"
        % (class_unhealthy_label, df_unhealthy.shape[0])
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel(f"Wave length of wavelet (in minute) (n_scales={len(scales)})")
    # ax4.set_yscale('log')

    ax3.set_yticks(np.arange(1, len(scales) + 1))
    ax3.set_yticklabels(
        ["%.1f" % x if i % 2 == 0 else "" for i, x in enumerate(scales)]
    )

    ax4.set_yticks(np.arange(1, len(scales) + 1))
    ax4.set_yticklabels(
        ["%.1f" % x if i % 2 == 0 else "" for i, x in enumerate(scales)]
    )
    # plt.show()
    filename = (
        f"{step_slug.replace('->', '_')}_{title.replace(' ', '_')}_{filename_sub}.png"
    )
    filepath = graph_outputdir / filename
    graph_outputdir.mkdir(parents=True, exist_ok=True)
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


def plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, classifier_name, fig):
    ax.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    lo, hi = mean_confidence_interval(aucs)
    label = r"Mean ROC (Mean AUC = %0.2f, 95%% CI [%0.4f, %0.4f] )" % (mean_auc, lo, hi)
    if len(aucs) <= 2:
        label = r"Mean ROC (Mean AUC = %0.2f)" % mean_auc
    ax.plot(mean_fpr, mean_tpr, color="tab:blue", label=label, lw=2, alpha=0.8)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic iteration",
    )
    ax.legend(loc="lower right")
    # fig.show()
    path = out_dir / "roc_curve" / "png"
    final_path = path / f"roc_{classifier_name}.png"
    print(final_path)
    final_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_path)

    path = out_dir / "roc_curve" / "svg"
    final_path = path / f"roc_{classifier_name}.svg"
    print(final_path)
    final_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_path)

    return mean_auc


def make_roc_curve(out_dir, classifier, X, y, cv, param_str, animal):
    # print("make_roc_curve")
    # details = str(classifier).replace("\n", "").replace(" ", "")
    # slug = "".join(x for x in details if x.isalnum())
    #
    # clf_name = "%s_%s" % (slug, param_str)
    # print(clf_name)
    #
    # if isinstance(X, pd.DataFrame):
    #     X = X.values
    #
    # tprs = []
    # aucs = []
    # mean_fpr = np.linspace(0, 1, 100)
    # plt.clf()
    # fig, ax = plt.subplots()
    # for i, (train, test) in enumerate(cv.split(X, y)):
    #     if isinstance(cv, RepeatedStratifiedKFold):
    #         print("make_roc_curve fold %d/%d" % (i, cv.get_n_splits()))
    #     else:
    #         print("make_roc_curve fold %d/%d" % (i, cv.nfold))
    #     classifier.fit(X[train], y[train])
    #     viz = plot_roc_curve(
    #         classifier,
    #         X[test],
    #         y[test],
    #         label=None,
    #         alpha=0.3,
    #         lw=1,
    #         ax=ax,
    #         c="tab:blue",
    #     )
    #     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    #     interp_tpr[0] = 0.0
    #     if np.isnan(viz.roc_auc):
    #         continue
    #     tprs.append(interp_tpr)
    #     aucs.append(viz.roc_auc)
    #     # ax.plot(viz.fpr, viz.tpr, c="tab:green")
    # print("make_roc_curve done!")
    # mean_auc = plot_roc_range(ax, tprs, mean_fpr, aucs, out_dir, clf_name, fig)
    # plt.close(fig)
    # plt.clf()
    return 0


def plot_dwt_power(
    epoch,
    date,
    animal_id,
    target,
    step_slug,
    out_dir,
    i,
    activity,
    dwt_scalogram,
    dwt_time,
    freqs,
    format_xaxis=False,
    vmin=None,
    vmax=None,
    standard_scale=False
):
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(19.2, 7.2))
    ticks = list(range(len(activity)))
    if format_xaxis:
        ticks = list(range(len(activity)))
    if standard_scale:
        axs[0].plot(activity, label="activity")
    else:
        axs[0].plot(ticks, activity, label="activity")

    axs[0].legend(loc="upper right")
    axs[0].set_title(
        "Time domain signal\n %s %s %s" % (date.replace("_", "/"), animal_id, str(target))
    )

    axs[0].set(xlabel="Time", ylabel="activity")
    if format_xaxis:
        axs[0].set(xlabel="Time", ylabel="activity")

    p = dwt_scalogram.copy()
    # if "ANSCOMBE" in step_slug:
    #     p = anscombe(p)

    #pos = axs[1].pcolormesh(dwt_time, freqs, np.sqrt(p), shading="gouraud")
    pos = axs[1].imshow(dwt_scalogram, interpolation='nearest', aspect="auto",
                        origin="lower", extent=[0, len(activity), 0, len(dwt_scalogram)])
    # pos = axs[1].imshow(np.sqrt(p), extent=[0, p.shape[1], p.shape[0], 1])
    fig.colorbar(pos, ax=axs[1])

    axs[1].set_aspect("auto")
    axs[1].set_title(f"DWT Coefficients {animal_id} {str(target)}")
    axs[1].set_xlabel("Time")
    if format_xaxis:
        axs[1].set_xlabel("Time")
    axs[1].set_ylabel("levels")
    #axs[1].set_yscale('log')
    freq_labels = [f"{x} ({pywt.scale2frequency('coif1', x)}Hz)" for x in list(range(1, dwt_scalogram.shape[0]+1))]
    axs[1].set_yticks(np.arange(0, len(freq_labels)))
    axs[1].set_yticklabels(freq_labels)

    if format_xaxis:
        n_x_ticks = axs[1].get_xticks().shape[0]
        labels_ = list(range(dwt_time.shape[1]))
        labels_ = np.array(labels_)[
            list(range(1, len(labels_), int(len(labels_) / n_x_ticks)))
        ]
        labels_[-2] = labels_[-1]
        axs[1].set_xticklabels(labels_)

    filename = f"{animal_id}_{str(target)}_{epoch}_{date}_idx_{i}_{step_slug}_dwt.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)


def plot_stft_power(
    sfft_window,
    stft_time,
    epoch,
    date,
    animal_id,
    target,
    step_slug,
    out_dir,
    i,
    activity,
    power_sfft,
    freqs,
    format_xaxis=False,
    vmin=None,
    vmax=None,
    standard_scale=False,
):
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(19.2, 7.2))
    ticks = list(range(len(activity)))
    if format_xaxis:
        ticks = get_time_ticks(len(activity))
    if standard_scale:
        axs[0].plot(activity, label="activity")
    else:
        axs[0].plot(ticks, activity, label="activity")
    # axs[0].plot(ticks, activity_centered,
    #             label='activity centered (signal - average of all sample (=%.2f))' % avg)
    axs[0].legend(loc="upper right")
    axs[0].set_title(
        "Time domain signal\n %s %s %s" % (date.replace("_", "/"), animal_id, str(target))
    )

    if standard_scale:
        if "STFT" in step_slug:
            axs[0].set_title(
                "Flatten STFT after Standard Scaling %s %s %s"
                % (date.replace("_", "/"), animal_id, str(target))
            )

    axs[0].set(xlabel="Time", ylabel="activity")
    if format_xaxis:
        axs[0].set(xlabel="Time", ylabel="activity")

    if standard_scale:
        axs[0].set(xlabel="Coefficients", ylabel="value")

    if format_xaxis:
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axs[0].xaxis.set_major_locator(mdates.DayLocator())

    p = power_sfft.copy()
    # if "ANSCOMBE" in step_slug:
    #     p = anscombe(p)

    pos = axs[1].pcolormesh(stft_time, freqs, np.sqrt(p), shading="gouraud")
    # pos = axs[1].imshow(np.sqrt(p), extent=[0, p.shape[1], p.shape[0], 1])
    fig.colorbar(pos, ax=axs[1])

    axs[1].set_aspect("auto")
    if sfft_window is None:
        sfft_window = 256  # default value
    axs[1].set_title(
        "STFT Power | window size=%d %s %s %s"
        % (sfft_window, date.replace("_", "/"), animal_id, str(target))
    )
    axs[1].set_xlabel("Time")
    if format_xaxis:
        axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Frequency (Hz)")
    # axs[1].set_yscale('log')

    if format_xaxis:
        n_x_ticks = axs[1].get_xticks().shape[0]
        labels_ = [item.strftime("%H:00") for item in ticks]
        labels_ = np.array(labels_)[
            list(range(1, len(labels_), int(len(labels_) / n_x_ticks)))
        ]
        labels_[:] = labels_[0]
        axs[1].set_xticklabels(labels_)

    filename = f"{animal_id}_{str(target)}_{epoch}_{date}_idx_{i}_{step_slug}_sfft.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)


def plot_cwt_power(
    vmin,
    vmax,
    epoch,
    date,
    animal_id,
    target,
    step_slug,
    out_dir,
    i,
    activity,
    power_masked,
    coi_line_array,
    scales,
    format_xaxis=True,
    standard_scale=False,
    wavelet=None,
    log_yaxis=False,
    filename_sub="power",
):
    plt.clf()
    if wavelet is not None:
        fig, axs = plt.subplots(1, 3, figsize=(19.2, 7.2))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(19.2, 7.2))

    # fig.suptitle("Signal , CWT", fontsize=18)
    ticks = list(range(len(activity)))
    if format_xaxis:
        ticks = get_time_ticks(len(activity))

    if standard_scale:
        axs[0].plot(activity, label="activity")
    else:
        axs[0].plot(ticks, activity, label="activity")

    # axs[0].plot(ticks, activity_centered,
    #             label='activity centered (signal - average of all sample (=%.2f))' % avg)
    axs[0].legend(loc="upper right")
    axs[0].set_title(
        "Time domain signal\n %s %s %s" % (date.replace("_", "/"), animal_id, str(target))
    )
    if standard_scale:
        if "CWT" in step_slug:
            axs[0].set_title(
                "Flatten CWT after Standard Scaling %s %s %s"
                % (date.replace("_", "/"), animal_id, str(target))
            )

    axs[0].set(xlabel="Time", ylabel="activity")

    if format_xaxis:
        axs[0].set(xlabel="Time", ylabel="activity")

    if standard_scale:
        axs[0].set(xlabel="Coefficients", ylabel="value")

    if format_xaxis:
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axs[0].xaxis.set_major_locator(mdates.DayLocator())

    p = power_masked.copy()
    if vmax is not None:
        pos = axs[1].imshow(
            p,
            extent=[0, p.shape[1], p.shape[0], 1],
            # vmin=vmin,
            # vmax=vmax,
            interpolation="nearest"
        )
    else:
        # p = StandardScaler(with_mean=False, with_std=True).fit_transform(p)
        pos = axs[1].imshow(
            p, extent=[0, p.shape[1], p.shape[0], 1], interpolation="nearest"
        )

    fig.colorbar(pos, ax=axs[1])

    axs[1].set_yticks(np.arange(1, len(scales) + 1))

    axs[1].set_yticklabels(
        ["%.1f" % x if i % 2 == 0 else "" for i, x in enumerate(scales)]
    )

    axs[1].set_aspect("auto")
    axs[1].set_title("CWT %s %s %s" % (date.replace("_", "/"), animal_id, str(target)))
    if activity is None:
        axs[1].set_title(
            "CWT after Standard Scaling %s %s %s"
            % (date.replace("_", "/"), animal_id, str(target))
        )

    axs[1].set_xlabel("Time")
    if format_xaxis:
        axs[1].set_xlabel("Time")
    axs[1].set_ylabel(f"Wave length of wavelet (in minute) (n_scales={len(scales)})")
    # if log_yaxis:
    #     axs[1].set_yscale('log')

    if format_xaxis:
        n_x_ticks = axs[1].get_xticks().shape[0]
        labels_ = [item.strftime("%H:00") for item in ticks]
        labels_ = np.array(labels_)[
            list(range(1, len(labels_), int(len(labels_) / n_x_ticks)))
        ]
        labels_[:] = labels_[0]
        axs[1].set_xticklabels(labels_)

    if wavelet is not None:
        wavelet_func = [wavelet.psi(x) for x in np.arange(-10, 10, 0.1)]
        axs[2].plot(wavelet_func, label="Real Component")
        axs[2].legend(loc="upper right")
        axs[2].set_title("%s wavelet function" % wavelet.name)
        axs[2].set(xlabel="Time (unit of time)", ylabel="amplitude")
    # axs[1].tick_params(axis='y', which='both', colors='black')
    filename = f"{animal_id}_{str(target)}_{epoch}_{date}_idx_{i}_{step_slug}_cwt_{filename_sub}_scales_{len(scales)}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)


def mask_cwt(cwt, coi):
    if coi is None:
        return cwt
    mask = np.ones(cwt.shape)
    for i in range(cwt.shape[1]):
        mask[int(coi[i]) :, i] = np.nan
    cwt = mask * cwt
    return cwt


def CWTVisualisation(
    N_META,
    step_slug,
    out_dir,
    shape,
    coi_mask,
    scales,
    coi_line_array,
    df_timedomain,
    df_cwt,
    class_healthy_label,
    class_unhealthy_label,
    filename_sub="power",
    plot_all_target = None
):
    print("CWTVisualisation")
    if plot_all_target is None:
        idx_healthy = df_timedomain[df_timedomain["health"] == 0].index.tolist()
        idx_unhealthy = df_timedomain[df_timedomain["health"] == 1].index.tolist()
        # coi_mask_ = coi_mask.astype(int)
        # idxs = np.where(coi_mask_ == 1)
        # df_cwt.columns = [str(x) for x in list(df_cwt.columns)]
        h_m = np.mean(df_cwt.loc[idx_healthy].values, axis=0).reshape(shape)
        uh_m = np.mean(df_cwt.loc[idx_unhealthy].values, axis=0).reshape(shape)
        plot_cwt_power_sidebyside(
            filename_sub,
            step_slug,
            True,
            class_healthy_label,
            class_unhealthy_label,
            idx_healthy,
            idx_unhealthy,
            coi_line_array,
            df_timedomain,
            out_dir,
            h_m,
            uh_m,
            scales,
            meta_size=N_META,
            ntraces=2,
        )
    else:
        for target in df_timedomain["target"].drop_duplicates().values:
            label = df_timedomain[df_timedomain["target"] == target]["label"].values[0]
            idx_target = df_timedomain[df_timedomain["target"] == target].index.tolist()
            cwt_m = np.mean(df_cwt.loc[idx_target].values, axis=0).reshape(shape)
            activity = np.mean(df_timedomain[df_timedomain["target"] == target].iloc[:, :-N_META].values, axis=0)
            plot_cwt_power(
                0,
                3,
                0,
                f"Element wise mean for label({label})",
                "",
                len(idx_target),
                step_slug,
                out_dir,
                0,
                activity,
                cwt_m.copy(),
                None,
                scales,
                log_yaxis=False,
                format_xaxis=False,
            )
        # stack cwt figures
        files = list(out_dir.glob("*.png"))
        concatenate_images(files, out_dir.parent.parent.parent, filename="cwt_elemwise_mean_per_label.png")


def check_scale_spacing(scales):
    spaces = []
    for i in range(scales.shape[0] - 1):
        wavelet_scale_space = scales[i] - scales[i + 1]
        spaces.append(wavelet_scale_space)
    print(spaces)
    print(np.min(spaces), np.max(spaces))
    return np.mean(spaces)


def simple_example():
    num_steps = 512
    x = np.arange(num_steps)
    y = np.sin(2 * np.pi * x / 32)
    delta_t = 1
    scales = np.arange(1, num_steps + 1)
    # pycwt
    w = wavelet.Morlet(0.1)
    freqs = 1 / (w.flambda() * scales)
    test = 1 / freqs

    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        y, delta_t, wavelet=w, freqs=freqs
    )
    coefs_cc = np.conj(coefs)
    power_cwt = np.real(np.multiply(coefs, coefs_cc))

    fig, axs = plt.subplots(1, 2, figsize=(19.2, 7.2))
    axs[0].plot(y, label="signal")
    axs[0].set(xlabel="Time", ylabel="amplitude")

    pos = axs[1].imshow(power_cwt, interpolation="nearest")
    fig.colorbar(pos, ax=axs[1])
    axs[1].set_aspect("auto")
    axs[1].set_title("CWT")
    axs[1].set_xlabel("Time")
    # axs[1].set_yscale('log')
    #fig.show()
    fig.clear()
    plt.close(fig)


def create_scale_array(size, m=2, last_scale=None):
    scales = []
    p = 1
    for i in range(size):
        p = p * m
        scales.append(p)
    if last_scale is not None:
        scales[-1] = last_scale
    return np.array(scales)


def even_list(n):
    result = [1]
    for num in range(2, n * 2 + 1, 2):
        result.append(num)
    del result[-1]
    return np.asarray(result, dtype=np.int32)


def compute_cwt_paper_sd(activity, scales):
    coefs, freqs = pywt.cwt(activity, scales, "coif1")
    return coefs, None, scales, freqs


def compute_cwt_paper_hd(activity, scales, wavelet_f0, step_slug):
    #print("compute_cwt...")
    w = wavelet.Morlet(wavelet_f0)
    if "MEXH" in step_slug:
        w = wavelet.MexicanHat()
    freqs = 1 / (w.flambda() * scales)
    delta_t = 1
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        activity, delta_t, wavelet=w, freqs=freqs
    )
    wavelet_type = w.name.lower()
    return coefs.astype(np.complex64), coi.astype(np.float16), scales, freqs, wavelet_type, delta_t


def compute_cwt_new(y):
    coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1)
    coi = np.interp(
        coi, (coi.min(), coi.max()), (0, len(scales))
    )  # todo fix weird hack
    return coefs, coi, scales, freqs


def compute_spectogram_matlab(activity, scales):
    mat_a = matlab.double(matlab.cell2mat(activity.tolist()))
    mat_scales = matlab.double(matlab.cell2mat(scales.astype(np.double).tolist()))
    mat_coefs, mat_freqs, _ = matlab.spectrogram(mat_a, mat_scales, nargout=3)
    coefs = np.asarray(mat_coefs)
    freqs = np.asarray(mat_freqs).flatten()
    scales = 1 / freqs
    return coefs, None, scales, freqs


def compute_cwt_matlab(activity, wavelet_name, scales):
    Fs = matlab.double(1 / 60)
    mat_scales = matlab.double(matlab.cell2mat(scales.astype(np.double).tolist()))
    mat_a = matlab.double(matlab.cell2mat(activity.tolist()))
    mat_coefs, mat_freqs = matlab.cwt(mat_a, mat_scales, wavelet_name, Fs, nargout=2)
    coefs = np.asarray(mat_coefs)
    freqs = np.asarray(mat_freqs).flatten()
    return coefs, None, scales, freqs


def compute_cwt_matlab_2(activity, wavelet_name):
    Fs = matlab.double(1 / 60)
    mat_a = matlab.double(matlab.cell2mat(activity.tolist()))
    mat_coefs, mat_freqs, mat_coi = matlab.cwt(mat_a, wavelet_name, Fs, nargout=3)
    coefs = np.asarray(mat_coefs)
    freqs = np.asarray(mat_freqs).flatten()
    coi = np.asarray(mat_coi).flatten()
    scales = 1 / freqs
    print("number of scales is %d" % len(scales))
    return coefs, coi, scales, freqs


def compute_multi_res(
    activity, animal_id, target, epoch, date, i, step_slug, out_dir, scales, wavelet_f0
):
    n = matlab.double(8)
    mat_a = matlab.double(matlab.cell2mat(activity.tolist()))
    mat_mra = matlab.modwtmra(matlab.modwt(mat_a, n))
    mra = np.asarray(mat_mra)
    plt.clf()
    fig, axs = plt.subplots(mra.shape[0], 2, figsize=(19.20, 19.20))
    for i in range(mra.shape[0]):
        coefs, coi, scales, freqs = compute_cwt_paper_hd(mra[i], scales, wavelet_f0)
        coefs_cc = np.conj(coefs)
        power_cwt = np.real(np.multiply(coefs, coefs_cc))

        pos = axs[i, 1].imshow(
            power_cwt,
            extent=[0, power_cwt.shape[1], power_cwt.shape[0], 1],
            interpolation="nearest",
        )
        fig.colorbar(pos, ax=axs[i, 1])
        axs[i, 1].set_aspect("auto")
        axs[i, 1].set_title("CWT %i" % i)
        axs[i, 1].set_xlabel("Time")
        axs[i, 1].set_ylabel("Wavelength")
        axs[i, 1].set_yscale("log")

        axs[i, 0].plot(mra[i])
        axs[i, 0].set_title("MRA decomposition %i" % i)
        axs[i, 0].set(xlabel="Time", ylabel="activity")

    filename = f"{animal_id}_{str(target)}_{epoch}_{date}_idx{i}_{step_slug}_mra.png"
    filepath = out_dir / filename
    filepath.mkdir(parents=True, exist_ok=True)
    print(filepath)
    fig.tight_layout()
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)
    return mra


def scalogram(y, coeffs):
    scalog = np.zeros((len(coeffs), len(coeffs[-1])*2))
    for i, c in enumerate(coeffs[::-1]):
        level_data = []
        for s in c:
            n_repeat = pow(2, i + 1)
            for _ in range(n_repeat):
                level_data.append(s)
        scalog[i, :] = level_data[0: scalog.shape[1]]
    scalog_rev = scalog[::-1]
    levels = list(range(1, scalog_rev.shape[0]+1))
    timeaxis = list(range(scalog_rev.shape[1]))
    return timeaxis, levels, scalog_rev


def dwt_power(
    activity,
    animal_id,
    target,
    date,
    epoch,
    sfft_window,
    enable_graph_out,
    step_slug,
    i,
    out_dir,
):
    w = pywt.Wavelet('coif1')
    mode = "symmetric"
    coeffs = pywt.wavedec(activity, w, mode=mode)
    wavelet = w.name

    dwt_time, levels, cc = scalogram(activity, coeffs)#ignore bottom 2 levels

    if enable_graph_out:
        plot_dwt_power(
            epoch,
            date,
            animal_id,
            target,
            step_slug,
            out_dir,
            i,
            activity,
            cc,
            dwt_time,
            levels,
        )
    # dwt_data = np.hstack(coeffs)
    dwt_data = np.hstack(cc).astype(np.float16)
    return dwt_data, dwt_data.shape, 1, dwt_data.shape[0], cc.real, cc.shape, wavelet, mode


def stft_power(
    activity,
    animal_id,
    target,
    date,
    epoch,
    sfft_window,
    enable_graph_out,
    step_slug,
    i,
    out_dir,
):
    freqs, stft_time, coefs = signal.stft(activity, fs=1, nperseg=sfft_window)
    coefs_cc = np.conj(coefs)
    power = np.real(np.multiply(coefs, coefs_cc))

    if enable_graph_out:
        plot_stft_power(
            sfft_window,
            stft_time,
            epoch,
            date,
            animal_id,
            target,
            step_slug,
            out_dir,
            i,
            activity,
            power.copy(),
            freqs,
        )
    return power, power.shape, stft_time, freqs


def cwt_power(
    hd,
    vmin,
    vmax,
    wavelet_f0,
    epoch,
    date,
    animal_id,
    target,
    activity,
    out_dir,
    i=0,
    step_slug="CWT_POWER",
    format_xaxis=None,
    avg=0,
    nscales=10,
    enable_graph_out=True,
    enable_coi=True,
    sub_sample_scales= 1
):
    scales = np.array([float(np.power(2, n)) for n in np.arange(1, nscales + 1, 0.1)])[::sub_sample_scales]

    coefs, coi, scales, freqs, wavelet_type, delta_t = compute_cwt_paper_hd(
        activity, scales, wavelet_f0, step_slug
    )
    coi = np.log(coi)
    coi = np.interp(
        coi, (coi.min(), coi.max()), (0, len(scales))
    )  # todo fix weird hack

    coefs_cc = np.conj(coefs)
    power_cwt = np.real(np.multiply(coefs, coefs_cc))
    # power_cwt = coefs.real
    if enable_coi:
        power_masked = mask_cwt(power_cwt.copy(), coi)
        power_cwt = power_masked
    else:
        power_masked = power_cwt.copy()

    imag = coefs.copy().imag.astype(np.float16)
    real = coefs.copy().real.astype(np.float16)
    mag = abs(coefs.copy()).astype(np.float16)
    if enable_coi:
        imag = mask_cwt(imag.copy(), coi)
        real = mask_cwt(real.copy(), coi)
    #cwt_raw = np.concatenate([real, imag])
    cwt_raw = mag
    power_cwt = cwt_raw
    power_masked = cwt_raw
    if enable_coi:
        power_masked = mask_cwt(power_masked.copy(), coi)

    if enable_graph_out:
        plot_cwt_power(
            vmin,
            vmax,
            epoch,
            date,
            animal_id,
            target,
            step_slug,
            out_dir,
            i,
            activity,
            cwt_raw.copy(),
            coi,
            scales,
            format_xaxis=format_xaxis,
            wavelet=None,
            log_yaxis=False,
        )

    return power_cwt, cwt_raw, freqs, coi, power_masked.shape, scales, wavelet_type, delta_t


def parse_param(animals_id, dates, i, targets, step_slug):
    if animals_id is None:
        return "Mean sample", step_slug, "", ""

    if len(animals_id) > 0:
        animal_id = str(int(float(animals_id[i])))
        target = targets[i]
        date = datetime.strptime(dates[i], "%d/%m/%Y").strftime("%d_%m_%Y")
        epoch = str(int(datetime.strptime(dates[i], "%d/%m/%Y").timestamp()))
        return animal_id, target, date, epoch
    else:
        return "Mean sample", step_slug, "", ""


def compute_cwt(
    hd,
    wavelet_f0,
    X,
    out_dir,
    step_slug,
    n_scales,
    animals_id,
    targets,
    dates,
    format_xaxis,
    vmin,
    vmax,
    enable_graph_out = None,
    sub_sample_scales = 1,
    enable_coi = True
):
    #print("compute_cwt...")
    out_dir = out_dir / "_cwt"
    # plotHeatmap(X, out_dir=out_dir, title="Time domain samples", force_xrange=True, filename="time_domain_samples.html")
    cwt = []
    cwt_raw = []
    # cwt_full = []
    i = 0
    std_scales = []
    for activity in tqdm(X):
        animal_id, target, date, epoch = parse_param(
            animals_id, dates, i, targets, step_slug
        )
        power, raw, freqs, coi, shape, scales, wavelet_type, delta_t = cwt_power(
            hd,
            vmin,
            vmax,
            wavelet_f0,
            epoch,
            date,
            animal_id,
            target,
            activity,
            out_dir,
            i,
            step_slug,
            format_xaxis,
            avg=np.average(X),
            nscales=n_scales,
            enable_coi=enable_coi,
            enable_graph_out=enable_graph_out,
            sub_sample_scales=sub_sample_scales
        )
        std_scales.append(np.nanstd(power, axis=1))
        power_flatten = np.array(power.flatten())
        # cwt_full.append(power_flatten_masked)
        coi_mask = np.isnan(power_flatten)
        # power_flatten_masked = power_flatten_masked[~np.isnan(power_flatten_masked)]  # remove masked values
        # power_flatten_masked_fft = np.concatenate([power_flatten_masked, power_fft])
        cwt.append(power_flatten)
        i += 1
    #print("convert cwt list to np array...")
    cwt = np.array(cwt)
    #print("done.")
    # cwt_full = np.array(cwt_full)
    std_scales = pd.DataFrame(np.array(std_scales))
    # plotHeatmap(cwt, out_dir=out_dir, title="CWT samples", force_xrange=True, filename="CWT.html", head=False)
    # plotHeatmap(cwt, out_dir=out_dir, title="CWT samples", force_xrange=True, filename="CWT_sub.html", head=True)
    return std_scales, cwt, cwt_raw, freqs, coi, shape, coi_mask, scales, wavelet_type, delta_t


def compute_dwt(X, animals_id, dates, step_slug, dwt_window, targets, out_dir,
            enable_graph_out=False):
    print("compute_dwt...")
    out_dir = out_dir / "_dwt"
    dwts = []
    dwts_cc = []
    i = 0
    for activity in tqdm(X):
        animal_id, target, date, epoch = parse_param(
            animals_id, dates, i, targets, step_slug
        )
        dwt, shape, dwt_time, freqs, cc, cc_shape, wavelet, mode = dwt_power(
            activity,
            animal_id,
            target,
            date,
            epoch,
            dwt_window,
            enable_graph_out,
            step_slug,
            i,
            out_dir,
        )
        dwts.append(dwt.flatten())
        dwts_cc.append(cc.flatten())
        i += 1
    dwts = np.array(dwts)
    dwts_cc = np.array(dwts_cc)

    return dwts, shape, dwts_cc, cc_shape, wavelet, mode


def compute_sfft(X, animals_id, dates, step_slug, sfft_window, targets, out_dir, enable_graph_out=False):
    print("compute_sfft...")
    out_dir = out_dir / "_stft"
    sffts = []
    i = 0
    for activity in tqdm(X):
        animal_id, target, date, epoch = parse_param(
            animals_id, dates, i, targets, step_slug
        )
        sfft, shape, stft_time, freqs = stft_power(
            activity,
            animal_id,
            target,
            date,
            epoch,
            sfft_window,
            enable_graph_out,
            step_slug,
            i,
            out_dir,
        )
        sffts.append(sfft.flatten())
        i += 1
    sffts = np.array(sffts)

    return sffts, shape, stft_time, freqs


class CWT(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        hd=False,
        wavelet_f0=None,
        out_dir=None,
        copy=True,
        step_slug=None,
        format_xaxis=False,
        n_scales=None,
        targets=None,
        animal_ids=None,
        dates=None,
        vmin=None,
        vmax=None,
        enable_graph_out=None,
        sub_sample_scales=None,
        enable_coi=True,
        delta_t=None,
        wavelet_type=None
    ):
        self.out_dir = out_dir
        self.copy = copy
        self.wavelet_f0 = wavelet_f0
        self.freqs = None
        self.coi = None
        self.shape = None
        self.n_scales = n_scales
        self.step_slug = step_slug
        self.format_xaxis = format_xaxis
        self.coi_mask = None
        self.animal_ids = animal_ids
        self.targets = targets
        self.dates = dates
        self.vmin = vmin
        self.vmax = vmax
        self.hd = hd
        self.scales = None
        self.enable_graph_out = enable_graph_out
        self.sub_sample_scales = sub_sample_scales
        self.enable_coi = enable_coi

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        # copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr')
        std_scales, cwt, cwt_raw, freqs, coi, shape, coi_mask, scales, wavelet_type, delta_t = compute_cwt(
            self.hd,
            self.wavelet_f0,
            X,
            self.out_dir,
            self.step_slug,
            self.n_scales,
            self.animal_ids,
            self.targets,
            self.dates,
            self.format_xaxis,
            self.vmin,
            self.vmax,
            self.enable_graph_out,
            self.sub_sample_scales,
            self.enable_coi
        )
        self.wavelet_type = wavelet_type
        self.delta_t = delta_t
        self.freqs = freqs
        self.coi = coi
        self.shape = shape
        self.coi_mask = coi_mask
        self.scales = scales
        return cwt, cwt_raw, std_scales

    def get_scales(self):
        return self.scales


class STFT(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        out_dir=None,
        copy=True,
        step_slug=None,
        animal_ids=None,
        dates=None,
        sfft_window=None,
        targets=None,
        enable_graph_out=False
    ):
        self.out_dir = out_dir
        self.copy = copy
        self.step_slug = step_slug
        self.animal_ids = animal_ids
        self.dates = dates
        self.step_slug = step_slug
        self.sfft_window = sfft_window
        self.targets = targets
        self.shape = None
        self.stft_time = None
        self.freqs = None
        self.enable_graph_out = enable_graph_out

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        # copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr')
        X, shape, stft_time, freqs = compute_sfft(
            X,
            self.animal_ids,
            self.dates,
            self.step_slug,
            self.sfft_window,
            self.targets,
            self.out_dir,
            enable_graph_out = self.enable_graph_out,
        )
        self.shape = shape
        self.stft_time = stft_time
        self.freqs = freqs
        return X


class DWT(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        out_dir=None,
        copy=True,
        step_slug=None,
        animal_ids=None,
        dates=None,
        dwt_window=None,
        targets=None,
        enable_graph_out=False,
        mode=None,
        wavelet=None
    ):
        self.out_dir = out_dir
        self.copy = copy
        self.step_slug = step_slug
        self.animal_ids = animal_ids
        self.dates = dates
        self.step_slug = step_slug
        self.dwt_window = dwt_window
        self.targets = targets
        self.shape = None
        self.dwt_time = None
        self.freqs = None
        self.enable_graph_out = enable_graph_out

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        # copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr')
        X, shape, X_cc, cc_shape, wavelet, mode = compute_dwt(
            X,
            self.animal_ids,
            self.dates,
            self.step_slug,
            self.dwt_window,
            self.targets,
            self.out_dir,
            enable_graph_out=self.enable_graph_out
        )
        self.shape = cc_shape
        self.mode = mode
        self.wavelet= wavelet
        return X, X_cc


def createSynthetic(activity):
    pure = activity
    noise = np.random.normal(0, 200, len(activity))
    signal = pure + noise
    synt = signal * np.random.uniform(0.1, 1.5)
    synt[synt < 0] = 0
    return synt.astype(int)


def create_synthetic_activity_data(n_samples=4):
    print("createSyntheticActivityData")
    samples_path = (
        "F:/Data2/dataset_gain_7day/activity_delmas_70101200027_dbft_7_1min.csv"
    )
    df = pd.read_csv(samples_path, header=None)
    df = df.fillna(0)
    crop = -4 - int(df.shape[1] / 1.1)
    activity = df.iloc[259, :-4].values.astype(int)

    dataset = []
    for j in range(n_samples):
        A = createSynthetic(activity)
        dataset.append(A)

    return dataset


def plot_line(X, out_dir, title="title", filename="file.html"):
    # fig = make_subplots(rows=len(transponders), cols=1)
    fig = make_subplots(rows=1, cols=1)
    for i, sample in enumerate(X):
        timestamp = get_time_ticks(len(sample))
        trace = go.Scatter(
            opacity=0.8,
            mode='lines',
            x=timestamp,
            y=sample,
        )
        fig.append_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    # create_rec_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    # fig.show()
    return trace, title


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    # datetext = [x.strftime('%H:%M') for x in date_list]
    return date_list


def plot_heatmap(
    X,
    out_dir="",
    title="Heatmap",
    filename="heatmap.html",
    force_xrange=False,
    head=False,
):
    # fig = make_subplots(rows=len(transponders), cols=1)
    if head and X.shape[0] > 4:
        X = X[[0, 1, -2, -1]]
    ticks = get_time_ticks(X.shape[1])
    if force_xrange:
        ticks = list(range(X.shape[1]))

    fig = make_subplots(rows=1, cols=1)
    trace = go.Heatmap(z=X, x=ticks, y=list(range(X.shape[0])), colorscale="Viridis")
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    # fig.show()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(file_path)
    return trace, title


def createSinWave(f, time):
    t = np.linspace(0, time, int(time))
    y = np.sin(2.0 * np.pi * t * f) * 100
    return y.astype(float)


def createPoisson(time, s):
    seed = np.ceil(random.random() * 2) + 1
    noise = np.random.poisson(int(seed), size=int(time))
    y = noise * s
    return y


def createNormal(f, time, s):
    seed = np.ceil(f * 2) + 1
    noise = np.random.normal(seed, size=int(time))
    y = noise * s
    return y


def createPoisonWaves(d, signal10, signal2):
    targets = []
    waves = []
    t = d
    for s in signal10:
        waves.append(createPoisson(t, s))
        targets.append(1)
    for s in signal2:
        waves.append(createPoisson(t, s))
        targets.append(2)
    waves = np.array(waves)
    return waves, targets


def createNormalWaves(d, signal10, signal2):
    targets = []
    waves = []
    t = d
    for s in signal10:
        waves.append(createNormal(random.random(), t, s))
        targets.append(1)
    for s in signal2:
        waves.append(createNormal(random.random(), t, s))
        targets.append(2)
    waves = np.array(waves)
    return waves, targets


def creatSin(freq, time):
    t = np.linspace(0, time, int(time))
    y = np.sin(2.0 * np.pi * t * freq)
    y[y < 0] = 0
    return y


if __name__ == "__main__":
    print("********CWT*********")
    # simple_example()
    from scipy.signal import chirp

    T = 5
    n = 5000
    t = np.linspace(0, T, n, endpoint=False)
    f0 = 1
    f1 = 50
    y = chirp(t, f0, T, f1, method='logarithmic')


    X = [y]
    for i in np.arange(5, 100, 5):
        X.append(creatSin(i, 1440))

    # X = np.array(createSyntheticActivityData())
    # X = np.array(X) - np.average(np.array(X))
    X_DWT = DWT(
        enable_graph_out=True,
        out_dir=Path("F:/Data2/_dwt_unit_before"),
        step_slug="UNIT TEST",
        animal_ids=[],
        targets=[],
        dates=[]
    ).transform(X)
    # X_SFFT = STFT(out_dir="F:/Data2/_cwt_unit_before", step_slug="UNIT TEST", animal_ids=[], targets=[], dates=[]).transform(X)
    exit()

    for d in [(60 * 60 * 24 * 1) / 60, (60 * 60 * 24 * 7) / 60]:

        signal10 = []
        for _ in range(60):
            signal10.append(creatSin(15 + np.random.random() / 100, d))
        signal2 = []
        for _ in range(60):
            signal2.append(creatSin(2 + np.random.random() / 100, d))

        for out_dir in [
            "F:/Data2/_cwt_debug_poisson_%d/" % d,
            "F:/Data2/_cwt_debug_normal_%d/" % d,
        ]:
            # X = np.array(createSyntheticActivityData()
            # X, targets = createSinWaves(d)
            # X, targets = createPoisonWaves(d)
            if "poisson" in out_dir:
                X, targets = createPoisonWaves(d, signal10, signal2)

            if "normal" in out_dir:
                X, targets = createNormalWaves(d, signal10, signal2)

            # X = pd.concat([pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X), pd.DataFrame(X)], axis=1)
            # X.columns = list(range(X.shape[1]))
            # plotLine(X, out_dir=out_dir, title="Activity samples", filename="X.html")

            X_CWT, X_REAL = CWT(
                wavelet_f0=1, out_dir=out_dir, format_xaxis=False
            ).transform(X)

            # plotLine(X_CWT, out_dir=out_dir, title="CWT samples", filename="CWT.html")
            print("********************")
            report_rows_list = []
            y = np.array(targets)
            y = y.astype(int)

            class_healthy = 1
            class_unhealthy = 2
            cross_validation_method = RepeatedStratifiedKFold(
                n_splits=10, n_repeats=10, random_state=0
            )
            scoring = {
                "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
                # 'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
                "precision_score0": make_scorer(
                    precision_score, average=None, labels=[class_healthy]
                ),
                "precision_score1": make_scorer(
                    precision_score, average=None, labels=[class_unhealthy]
                ),
                "recall_score0": make_scorer(
                    recall_score, average=None, labels=[class_healthy]
                ),
                "recall_score1": make_scorer(
                    recall_score, average=None, labels=[class_unhealthy]
                ),
                "f1_score0": make_scorer(
                    f1_score, average=None, labels=[class_healthy]
                ),
                "f1_score1": make_scorer(
                    f1_score, average=None, labels=[class_unhealthy]
                ),
            }

            print("TimeDom->SVC")
            clf_svc = make_pipeline(SVC(probability=True, class_weight="balanced"))
            scores = cross_validate(
                clf_svc,
                X.copy(),
                y.copy(),
                cv=cross_validation_method,
                scoring=scoring,
                n_jobs=-1,
            )
            scores["class0"] = y[y == class_healthy].size
            scores["class1"] = y[y == class_unhealthy].size
            scores["option"] = "TimeDom"
            scores["days"] = d
            scores["farm_id"] = 0
            scores["balanced_accuracy_score_mean"] = np.mean(
                scores["test_balanced_accuracy_score"]
            )
            scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
            scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
            scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
            scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
            scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
            scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
            scores["classifier"] = "->SVC"
            scores["classifier_details"] = (
                str(clf_svc).replace("\n", "").replace(" ", "")
            )
            clf_svc = make_pipeline(SVC(probability=True, class_weight="balanced"))
            aucs = make_roc_curve(
                out_dir,
                clf_svc,
                X.copy(),
                y.copy(),
                cross_validation_method,
                "time_dom",
            )
            scores["roc_auc_score_mean"] = aucs
            print(aucs)
            report_rows_list.append(scores)
            del scores

            print("CWT->SVC")
            clf_svc = make_pipeline(SVC(probability=True, class_weight="balanced"))
            scores = cross_validate(
                clf_svc,
                X_CWT.copy(),
                y.copy(),
                cv=cross_validation_method,
                scoring=scoring,
                n_jobs=-1,
            )
            scores["class0"] = y[y == class_healthy].size
            scores["class1"] = y[y == class_unhealthy].size
            scores["option"] = "CWT"
            scores["days"] = d
            scores["farm_id"] = 0
            scores["balanced_accuracy_score_mean"] = np.mean(
                scores["test_balanced_accuracy_score"]
            )
            scores["precision_score0_mean"] = np.mean(scores["test_precision_score0"])
            scores["precision_score1_mean"] = np.mean(scores["test_precision_score1"])
            scores["recall_score0_mean"] = np.mean(scores["test_recall_score0"])
            scores["recall_score1_mean"] = np.mean(scores["test_recall_score1"])
            scores["f1_score0_mean"] = np.mean(scores["test_f1_score0"])
            scores["f1_score1_mean"] = np.mean(scores["test_f1_score1"])
            scores["classifier"] = "->SVC"
            scores["classifier_details"] = (
                str(clf_svc).replace("\n", "").replace(" ", "")
            )
            clf_svc = make_pipeline(SVC(probability=True, class_weight="balanced"))
            aucs = make_roc_curve(
                out_dir,
                clf_svc,
                X_CWT.copy(),
                y.copy(),
                cross_validation_method,
                "freq_dom",
            )
            scores["roc_auc_score_mean"] = aucs
            print(aucs)
            report_rows_list.append(scores)
            del scores

            df_report = pd.DataFrame(report_rows_list)

            if not os.path.exists(out_dir):
                print("mkdir", out_dir)
                os.makedirs(out_dir)
            filename = "%s/report.csv" % (out_dir)
            df_report.to_csv(filename, sep=",", index=False)
            print("filename=", filename)

            print("REPORT")
            df = pd.read_csv(str(filename), index_col=None)
            df["class_0_label"] = "1"
            df["class_1_label"] = "2"
            df["config"] = [format(str(x)) for x in list(zip(df.option, df.classifier))]
            df = df.sort_values("roc_auc_score_mean")

            print(df)

            t4 = "AUC performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
                df["days"].values[0],
                df["class0"].values[0],
                df["class_0_label"].values[0],
                df["class1"].values[0],
                df["class_1_label"].values[0],
            )

            t3 = "Accuracy performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
                df["days"].values[0],
                df["class0"].values[0],
                df["class_0_label"].values[0],
                df["class1"].values[0],
                df["class_1_label"].values[0],
            )

            t1 = "Precision class0 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
                df["days"].values[0],
                df["class0"].values[0],
                df["class_0_label"].values[0],
                df["class1"].values[0],
                df["class_1_label"].values[0],
            )

            t2 = "Precision class1 performance of different inputs<br>Days=%d class0=%d %s class1=%d %s" % (
                df["days"].values[0],
                df["class0"].values[0],
                df["class_0_label"].values[0],
                df["class1"].values[0],
                df["class_1_label"].values[0],
            )

            fig = make_subplots(rows=4, cols=1, subplot_titles=(t1, t2, t3, t4))

            fig.append_trace(
                px.bar(df, x="config", y="precision_score0_mean").data[0], row=1, col=1
            )
            fig.append_trace(
                px.bar(df, x="config", y="precision_score1_mean").data[0], row=2, col=1
            )
            fig.append_trace(
                px.bar(df, x="config", y="balanced_accuracy_score_mean").data[0],
                row=3,
                col=1,
            )
            fig.append_trace(
                px.bar(df, x="config", y="roc_auc_score_mean").data[0], row=4, col=1
            )

            fig.update_yaxes(range=[0, 1], row=1, col=1)
            fig.update_yaxes(range=[0, 1], row=2, col=1)
            fig.update_yaxes(range=[0, 1], row=3, col=1)
            fig.update_yaxes(range=[0, 1], row=4, col=1)

            fig.add_shape(
                type="line",
                x0=-0.0,
                y0=0.920,
                x1=1.0,
                y1=0.920,
                line=dict(
                    color="LightSeaGreen",
                    width=4,
                    dash="dot",
                ),
            )

            fig.add_shape(
                type="line",
                x0=-0.0,
                y0=0.640,
                x1=1.0,
                y1=0.640,
                line=dict(
                    color="LightSeaGreen",
                    width=4,
                    dash="dot",
                ),
            )

            fig.add_shape(
                type="line",
                x0=-0.0,
                y0=0.357,
                x1=1.0,
                y1=0.357,
                line=dict(
                    color="LightSeaGreen",
                    width=4,
                    dash="dot",
                ),
            )

            fig.add_shape(
                type="line",
                x0=-0.0,
                y0=0.078,
                x1=1.0,
                y1=0.078,
                line=dict(
                    color="LightSeaGreen",
                    width=4,
                    dash="dot",
                ),
            )

            fig.update_xaxes(showticklabels=False)  # hide all the xticks
            fig.update_xaxes(showticklabels=True, row=4, col=1)

            fig.update_yaxes(showgrid=True, gridwidth=1)
            fig.update_xaxes(showgrid=True, gridwidth=1)

            filename = "%s/ML_performance.html" % (out_dir)
            print(filename)
            fig.write_html(filename)
            # fig.show()
