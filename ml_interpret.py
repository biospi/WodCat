from pathlib import Path
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt as wavelet
import typer
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from cwt._cwt import CWT, DWT
from model.data_loader import load_activity_data
from preprocessing.preprocessing import apply_preprocessing_steps
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def report_mannwhitney(array1, array2):
    U1, p = mannwhitneyu(list(array1), list(array2), method="exact")
    nx, ny = len(array1), len(array2)
    U2 = nx * ny - U1
    print(U2, U1, p)
    res = f"p={p:.3f}"
    print(res)
    return res


def elemwise_cwt(X, preprocessing_steps, output_dir):
    f_transform = CWT(
        step_slug="_".join(preprocessing_steps),
        wavelet_f0=6,
        out_dir=output_dir,
        n_scales=8,
        sub_sample_scales=1,
        enable_coi=False,
        enable_graph_out=False,
    )
    X_cwt, _, _ = f_transform.transform(X)

    cwt_list_0 = []
    for cwt in X_cwt:
        cwt_list_0.append(np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1])))
    coefs_class0_mean = np.mean(cwt_list_0, axis=0)
    return coefs_class0_mean


def get_imp_stat_cat(
    window_size,
    farmname,
    preprocessing_steps,
    date_list,
    imp,
    activity,
    X_train,
    output_dir,
    n_peaks=1
):
    # print(date_list)
    half_peaklenght = int(window_size/2)
    days_median = []
    nights_median = []
    days_std = []
    nights_std = []
    all_days = []
    all_nights = []
    idxs = []

    for i in range(n_peaks):
        start = i * half_peaklenght
        end = start + half_peaklenght
        a = activity[start:end]

        imp_ = imp[start:end]
        date = date_list[start:end]

        mask = np.array([0 if x < len(a)/2 else 1 for x in range(len(a))])

        light = np.zeros(mask.shape).astype(str)
        light[mask == 0] = "before"
        light[mask == 1] = "after"

        day_imp = imp_[light == "before"]
        night_imp = imp_[light == "after"]
        all_days.extend(day_imp)
        all_nights.extend(night_imp)
        # report_mannwhitney(day_imp, night_imp)

        std = [day_imp.std(), night_imp.std()]
        mean = [day_imp.mean(), night_imp.mean()]
        median = [np.median(day_imp), np.median(night_imp)]

        days_median.append(np.median(day_imp))
        nights_median.append(np.median(night_imp))

        days_std.append(np.std(day_imp))
        nights_std.append(np.std(night_imp))
        index = f"Peak {i}"
        idxs.append(index)

    # dfs.append(df)
    df = pd.DataFrame(
        {
            "before peak median": nights_median,
            "after peak median": days_median,
            "before peak std": nights_std,
            "after peak std": days_std,
        },
        index=idxs,
    )
    # df = pd.concat(dfs)
    fig_ = df.plot.barh(
        rot=0, title="Feature importance for each", figsize=(6, 6)
    ).get_figure()
    ax = fig_.gca()
    for j in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[j].set_weight("bold")
    ax.set_xlabel("Median of feature importance")
    ax.legend(loc="upper right")
    # for j in range(n_activity_days*2):
    #     if j % 2 == 0:
    #         continue
    #     ax.get_yticklabels()[j].set_weight("bold")

    filename = f"{-1}_per_{X_train.shape[1]}.png"
    filepath = output_dir / filename
    print(filepath)
    fig_.tight_layout()
    fig_.savefig(filepath)

    # dot plot
    plt.clf()
    plt.cla()
    df_ = pd.DataFrame({"before": days_median, "after": nights_median})
    fig_box = df_.boxplot().get_figure()
    ax = fig_box.gca()
    ax.set_title(
        f"Feature importance before vs after\n{report_mannwhitney(days_median, nights_median)}"
    )
    ax.set_ylabel("Mean of feature importance")
    for i, d in enumerate(df_):
        y = df_[d]
        x = np.random.normal(i + 1, 0.04, len(y))
        ax.plot(x, y, marker="o", linestyle="None", mfc="none")
    steps = "_".join(preprocessing_steps).lower()
    filename = f"{farmname}_{-1}_box_{X_train.shape[1]}_{steps}.png"
    filepath = output_dir / filename
    print(filepath)
    fig_box.set_size_inches(4, 4)
    fig_box.tight_layout()
    fig_box.savefig(filepath, dpi=500)

    mask = np.array([1 for _ in range(len(activity))])
    cpt = 0
    for n, v in enumerate(mask):
        if cpt < half_peaklenght:
            mask[n] = 0
        if cpt > half_peaklenght*2:
            cpt = 0
        cpt+=1


    light = np.zeros(mask.shape).astype(str)
    light[mask == 0] = "before"
    light[mask == 1] = "after"

    fig_, ax_ = plt.subplots(figsize=(8, 8))
    filename = f"{-1}_period_{X_train.shape[1]}.png"
    filepath = output_dir / filename
    print(filepath)
    ax_.plot(activity)
    ax_.plot(mask)
    fig_.savefig(filepath)

    day_imp = imp[light == "before"]
    night_imp = imp[light == "after"]

    std = [day_imp.std(), night_imp.std()]
    mean = [day_imp.mean(), night_imp.mean()]
    median = [np.median(day_imp), np.median(night_imp)]
    index = ["before peak", "after peak"]
    df = pd.DataFrame({"Std": std, "Mean": mean, "Median": median}, index=index)
    fig_ = df.plot.barh(rot=0, title="Feature importance", figsize=(8, 8)).get_figure()
    ax = fig_.gca()
    ax.set_xlabel("Feature importance")
    filename = f"{farmname}_{-1}_{X_train.shape[1]}_{steps}.png"
    filepath = output_dir / filename
    print(filepath)
    fig_.tight_layout()
    fig_.savefig(filepath)


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    class_healthy_label: List[str] = ["1To1"],
    class_unhealthy_label: List[str] = ["2To2"],
    meta_columns: List[str] = [
        "label",
        "id",
        "imputed_days",
        "date",
        "health",
        "target",
    ],
    preprocessing_steps: List[str] = ["L1", "ANSCOMBE", "LOG"],
    r_avg: int = 60,
    prct: int = 90,
    window_size:int = 15,
    transform: str = "cwt",
    enable_graph_out: bool = True,
    individual_to_ignore: List[str] = [],
    farmname: str = '',
    width: int = 9,
    height: int = 4,
    n_peaks=1
):
    """This script builds the graphs for cwt interpretation\n
    Args:\n
        output_dir: Output directory
        dataset_folder: Dataset input directory
        class_healthy: Label for healthy class
        class_unhealthy: Label for unhealthy class
        p: analyse famacha impact over time up to test date
    """

    # days, farm_id, option, sampling = parse_param_from_filename(file)
    print(f"loading dataset file {dataset_file} ...")
    (
        data_frame,
        df_meta,
        _,
        _,
        label_series,
        samples
    ) = load_activity_data(
        output_dir,
        meta_columns,
        dataset_file,
        class_healthy_label,
        class_unhealthy_label,
        individual_to_ignore=individual_to_ignore,
        preprocessing_steps=[]
    )

    data_frame_time, _, _ = apply_preprocessing_steps(
        meta_columns,
        None,
        None,
        None,
        None,
        data_frame.copy(),
        output_dir,
        preprocessing_steps,
        class_healthy_label,
        class_unhealthy_label,
        clf_name="SVM",
        keep_meta=True,
        output_qn_graph=False
    )

    data_frame_time = data_frame_time.loc[data_frame_time["health"].isin([0, 1])]

    X_train, y_train = (
        data_frame_time.iloc[:, :-len(meta_columns)].values,
        data_frame_time["health"].values,
    )

    clf = SVC(kernel="linear", probability=True)
    #clf = LinearRegression()
    #clf = LogisticRegression(n_jobs=-1)
    print("fit...")
    clf.fit(X_train, y_train)
    imp = abs(clf.coef_[0])

    mean_time = np.mean(X_train, axis=0)
    date_list = list(range(0, len(mean_time)))

    get_imp_stat_cat(
        window_size,
        farmname,
        preprocessing_steps,
        date_list,
        imp,
        mean_time,
        X_train,
        output_dir,
        n_peaks=n_peaks
    )

    fig, ax = plt.subplots(figsize=(width, height))
    ax2 = ax.twinx()
    ax.plot(
        date_list,
        mean_time,
        label=f"Mean activity (all samples)",
    )

    df_imp = pd.DataFrame(imp, columns=["imp"])
    roll_avg = df_imp.imp.rolling(r_avg).mean()
    n_b = len(roll_avg)
    roll_avg = roll_avg.dropna()
    n_a = len(roll_avg)
    pad = int(np.ceil((n_b - n_a)/2))
    roll_avg = [np.nan]*pad + roll_avg.to_list() + [np.nan]*pad
    roll_avg = np.array(roll_avg[0:len(date_list)])
    ax2.plot(
        date_list,
        roll_avg,
        color="black",
        label=f"Feature importance (roll avg, {r_avg})",
        alpha=0.9,
    )

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title(
        f"Feature importance {type(clf).__name__} n_peaks={n_peaks}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax2.set_ylabel("Absolute value of Coefficients")
    filename = f"{n_peaks}_feature_importance_{X_train.shape[1]}.png"
    filepath = output_dir / filename
    print(filepath)

    fig.savefig(filepath)

    if transform == "dwt":
        f_transform = DWT(
            dwt_window="coif1",
            step_slug="_".join(preprocessing_steps),
            out_dir=output_dir,
            enable_graph_out=enable_graph_out,
        )
        _, X_train = f_transform.transform(X_train)
    if transform == "cwt":
        f_transform = CWT(
            step_slug="_".join(preprocessing_steps),
            wavelet_f0=6,
            out_dir=output_dir,
            n_scales=8,
            sub_sample_scales=1,
            enable_coi=False,
            enable_graph_out=enable_graph_out,
        )
        X_train, _, _ = f_transform.transform(X_train)
    X_train_o = X_train.copy()
    X_train[np.isnan(X_train)] = -1
    # scales = CWT_Transform.get_scales()
    clf = SVC(kernel="linear", probability=True)
    # clf = LinearRegression()

    print("fit...")
    clf.fit(X_train, y_train)
    imp = abs(clf.coef_[0])

    intercept = clf.intercept_
    imp_top_n_perct = imp.copy()
    imp_top_n_perct[
        imp_top_n_perct <= np.percentile(imp_top_n_perct, prct)
    ] = np.nan

    mean_ = np.mean(X_train, axis=0)

    cwt_0 = X_train[y_train == 0]
    cwt_1 = X_train[y_train == 1]

    cwt_list_0 = []
    for cwt in cwt_0:
        cwt_list_0.append(
            np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
        )

    cwt_list_1 = []
    for cwt in cwt_1:
        cwt_list_1.append(
            np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
        )

    coefs_class0_mean = np.mean(cwt_list_0, axis=0)
    coefs_class1_mean = np.mean(cwt_list_1, axis=0)
    cwt_imp = np.reshape(imp, (f_transform.shape[0], f_transform.shape[1]))
    cwt_imp_top = np.reshape(
        imp_top_n_perct, (f_transform.shape[0], f_transform.shape[1])
    )

    fig, axs = plt.subplots(3, 1, facecolor="white", figsize=(12.80, 18.80))
    axs = axs.ravel()
    fig, ax = plt.subplots(figsize=(12.80, 7.20))
    ax2 = ax.twinx()
    ax.plot(mean_, label=f"mean {transform}(flatten) of all samples")
    # ax.plot(imp*mean, label="mean activity of all samples * feature importance")
    ax2.plot(imp, color="red", label="feature importance", alpha=0.3)
    df_imp = pd.DataFrame(imp, columns=["imp"])
    roll_avg = df_imp.imp.rolling(1000).mean()
    ax2.plot(
        roll_avg,
        color="black",
        label=f"feature importance rolling avg ({1000} points)",
        alpha=0.9,
    )

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title(f"Feature importance {type(clf).__name__} n_peaks={n_peaks}")
    ax.set_xlabel(f"{transform} (features)")
    ax.set_ylabel("Activity")
    ax2.set_ylabel("Absolute value of Coefficients", color="red")
    filename = (
        f"{n_peaks}_{transform}_feature_importance_{X_train.shape[1]}.png"
    )
    filepath = output_dir / filename
    print(filepath)
    fig.savefig(filepath)

    cwt_0 = X_train[y_train == 0]
    cwt_1 = X_train[y_train == 1]

    cwt_list_0 = []
    for cwt in cwt_0:
        # iwave_test = wavelet.icwt(np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1])), f_transform.scales, f_transform.delta_t,
        #                        wavelet=f_transform.wavelet_type.lower()).real
        # plt.plot(iwave_test)
        # plt.show()

        cwt_list_0.append(
            np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
        )

    cwt_list_1 = []
    for cwt in cwt_1:
        cwt_list_1.append(
            np.reshape(cwt, (f_transform.shape[0], f_transform.shape[1]))
        )

    coi_mask = np.reshape(
        X_train_o[0], (f_transform.shape[0], f_transform.shape[1])
    )
    coefs_class0_mean = np.mean(cwt_list_0, axis=0)
    coefs_class0_mean[np.isnan(coi_mask)] = np.nan
    coefs_class1_mean = np.mean(cwt_list_1, axis=0)
    coefs_class1_mean[np.isnan(coi_mask)] = np.nan
    cwt_imp = np.reshape(imp, (f_transform.shape[0], f_transform.shape[1]))
    # cwt_intercept = np.reshape(intercept, (f_transform.shape[0], f_transform.shape[1]))

    # cwt_imp[np.isnan(coi_mask)] = np.nan
    cwt_imp_top = np.reshape(
        imp_top_n_perct, (f_transform.shape[0], f_transform.shape[1])
    )
    cwt_imp_top[np.isnan(coi_mask)] = np.nan

    fig, axs = plt.subplots(5, 2, facecolor="white", figsize=(28.60, 26.80))
    origin = "upper"
    if transform == "dwt":
        fig, axs = plt.subplots(3, 2, facecolor="white", figsize=(28.60, 12.80))
        origin = "lower"
    axs = axs.ravel()

    # axs[0].pcolormesh(
    #     np.arange(coefs_class0_mean.shape[1]),
    #     scales,
    #     coefs_class0_mean,
    #     cmap="viridis"
    # )
    mat_max = max([np.nanmax(coefs_class0_mean), np.nanmax(coefs_class1_mean)])
    mat_min = min([np.nanmin(coefs_class0_mean), np.nanmin(coefs_class1_mean)])
    date_list = mdates.date2num(date_list)
    im = axs[0].imshow(
        coefs_class0_mean,
        origin=origin,
        extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
        interpolation="nearest",
        aspect="auto",
        vmin=mat_min,
        vmax=mat_max,
    )
    fig.colorbar(im, ax=axs[0])

    axs[0].set_title(f"Element wise mean of {transform} coefficients healthy")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Scales")

    im = axs[1].imshow(
        coefs_class1_mean,
        origin=origin,
        extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
        interpolation="nearest",
        aspect="auto",
        vmin=mat_min,
        vmax=mat_max,
    )
    fig.colorbar(im, ax=axs[1])

    axs[1].set_title(f"Element wise mean of {transform} coefficients unhealthy")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Scales")

    mat_max = max([np.nanmax(cwt_imp), np.nanmax(cwt_imp_top)])
    mat_min = min([np.nanmin(cwt_imp), np.nanmin(cwt_imp_top)])
    im = axs[2].imshow(
        cwt_imp,
        origin=origin,
        extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
        interpolation="nearest",
        aspect="auto",
        vmin=mat_min,
        vmax=mat_max,
    )
    fig.colorbar(im, ax=axs[2])

    axs[2].set_title(f"{transform} Features importance {type(clf).__name__}")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Scales")

    im = axs[3].imshow(
        cwt_imp_top,
        origin=origin,
        extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
        interpolation="nearest",
        aspect="auto",
        vmin=mat_min,
        vmax=mat_max,
    )
    fig.colorbar(im, ax=axs[3])

    axs[3].set_title(
        f"{transform} Features importance top 10% {type(clf).__name__} n_peaks={n_peaks}"
    )
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Scales")

    a = (cwt_imp * coefs_class0_mean) - intercept
    b = (cwt_imp * coefs_class1_mean) - intercept
    # if distance:
    #     a = (cwt_imp * coefs_class0_mean)
    #     b = (cwt_imp * coefs_class1_mean)

    mat_max = max([np.nanmax(a), np.nanmax(b)])
    mat_min = min([np.nanmin(a), np.nanmin(b)])
    im = axs[4].imshow(
        a,
        origin=origin,
        extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
        interpolation="nearest",
        aspect="auto",
        vmin=mat_min,
        vmax=mat_max,
    )
    fig.colorbar(im, ax=axs[4])

    axs[4].set_title(
        f"{transform} Features importance multipied by coef of healthy class n_peaks={n_peaks}"
    )
    axs[4].set_xlabel("Time")
    axs[4].set_ylabel("Scales")

    im = axs[5].imshow(
        b,
        origin=origin,
        extent=[date_list[0], date_list[-1], 1, coefs_class0_mean.shape[0]],
        interpolation="nearest",
        aspect="auto",
        vmin=mat_min,
        vmax=mat_max,
    )
    fig.colorbar(im, ax=axs[5])

    axs[5].set_title(
        f"{transform} Features importance multipied by coef of unhealthy class n_peaks={n_peaks}"
    )
    axs[5].set_xlabel("Time")
    axs[5].set_ylabel("Scales")
    #########################################

    if transform == "cwt":
        iwave_h = wavelet.icwt(
            coefs_class0_mean,
            f_transform.scales,
            f_transform.delta_t,
            wavelet=f_transform.wavelet_type.lower(),
        ).real
        iwave_uh = wavelet.icwt(
            coefs_class1_mean,
            f_transform.scales,
            f_transform.delta_t,
            wavelet=f_transform.wavelet_type.lower(),
        ).real
        ymin = min([iwave_h.min(), iwave_uh.min()])
        ymax = max([iwave_h.max(), iwave_uh.max()])

        axs[6].plot(iwave_h)
        fig.colorbar(im, ax=axs[6])
        # axs[6].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        # axs[6].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        axs[6].set_title(
            f"{transform} Inverse of coefs of healthy n_peaks={n_peaks}"
        )
        axs[6].set_xlabel("Time")
        axs[6].set_ylabel("Activity")
        axs[6].set_ylim([ymin, ymax])

        axs[7].plot(iwave_uh)
        fig.colorbar(im, ax=axs[7])
        # axs[7].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        # axs[7].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        axs[7].set_title(
            f"{transform} Inverse of coefs of healthy n_peaks={n_peaks}"
        )
        axs[7].set_xlabel("Time")
        axs[7].set_ylabel("Activity")
        axs[7].set_ylim([ymin, ymax])

        iwave_h = abs(
            wavelet.icwt(
                a,
                f_transform.scales,
                f_transform.delta_t,
                wavelet=f_transform.wavelet_type.lower(),
            )
        )
        iwave_uh = abs(
            wavelet.icwt(
                b,
                f_transform.scales,
                f_transform.delta_t,
                wavelet=f_transform.wavelet_type.lower(),
            )
        )
        ymin = min([iwave_h.min(), iwave_uh.min()])
        ymax = max([iwave_h.max(), iwave_uh.max()])

        axs[8].plot(iwave_h)
        fig.colorbar(im, ax=axs[8])
        # axs[8].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        # axs[8].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        axs[8].set_title(
            f"{transform} Inverse of Features importance multipied by coef of healthy n_peaks={n_peaks}"
        )
        axs[8].set_xlabel("Time")
        axs[8].set_ylabel("Activity")
        axs[8].set_ylim([ymin, ymax])

        axs[9].plot(iwave_uh)
        fig.colorbar(im, ax=axs[9])
        # axs[9].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        # axs[9].xaxis.set_major_locator(mdates.MinuteLocator(interval=T * n_activity_days))
        axs[9].set_title(
            f"{transform} Inverse of Features importance multipied by coef of healthy n_peaks={n_peaks}"
        )
        axs[9].set_xlabel("Time")
        axs[9].set_ylabel("Activity")
        axs[9].set_ylim([ymin, ymax])

    filename = f"{n_peaks}_{transform}_reshaped_feature_importance_{X_train.shape[1]}.png"
    filepath = output_dir / filename
    # fig.autofmt_xdate()
    fig.tight_layout()
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    # typer.run(main)
    output_dir = Path(f"E:/Cats/interpret")


    dataset = Path(
            "E:/Cats/paper_debug_regularisation_8/All_100_10_120_001/dataset/samples.csv"
        )
    meta_columns_file = dataset.parent / "meta_columns.csv"
    meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
    n_peak = 1

    main(
       output_dir,
        dataset,
        window_size=120,
        transform="dwt",
        enable_graph_out=False,
        farmname='cats',
        meta_columns=meta_columns,
        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        class_healthy_label=["0.0"],
        class_unhealthy_label=["1.0"],
        r_avg = 15,
        n_peaks=n_peak
    )



    dataset = Path(
            "E:/Cats/paper_debug_regularisation_8/All_100_10_120_002/dataset/samples.csv"
        )
    meta_columns_file = dataset.parent / "meta_columns.csv"
    meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
    n_peak = 2

    main(
       output_dir,
        dataset,
        window_size=120,
        transform="dwt",
        enable_graph_out=False,
        farmname='cats',
        meta_columns=meta_columns,
        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        class_healthy_label=["0.0"],
        class_unhealthy_label=["1.0"],
        r_avg = 15,
        n_peaks=n_peak
    )


    dataset = Path(
            "E:/Cats/paper_debug_regularisation_8/All_100_10_120_003/dataset/samples.csv"
        )
    meta_columns_file = dataset.parent / "meta_columns.csv"
    meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
    n_peak = 3

    main(
       output_dir,
        dataset,
        window_size=120,
        transform="dwt",
        enable_graph_out=False,
        farmname='cats',
        meta_columns=meta_columns,
        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        class_healthy_label=["0.0"],
        class_unhealthy_label=["1.0"],
        r_avg = 15,
        n_peaks=n_peak
    )

