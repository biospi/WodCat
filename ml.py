#
# Author: Axel Montout <axel.montout <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of PredictionOfHelminthsInfection.
#
# PHI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PHI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seaMass.  If not, see <http://www.gnu.org/licenses/>.
#
from pathlib import Path, PurePath
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from model.data_loader import load_activity_data
from model.svm import process_ml
from preprocessing.preprocessing import apply_preprocessing_steps
from utils.visualisation import (
    plot_mean_groups,
    plot_time_pca,
    plot_time_lda,
    plot_umap,
    plot_time_pls, plot_zeros_distrib, plot_groups, plot_crepuscular,
)


def build_hpc_string(
    study_id,
    output_dir,
    dataset_folder,
    preprocessing_steps,
    class_healthy_label,
    class_unhealthy_label,
    classifiers,
    pre_visu,
    n_job,
    skip,
    meta_columns,
    cv,
    individual_to_ignore,
    c,
    gamma
):

    o_d = Path("/user/work/fo18103") / output_dir.as_posix().split(':')[1][1:]
    d_d = Path("/user/work/fo18103") / dataset_folder.as_posix().split(':')[1][1:]
    # data_dir = str(dataset_folder).replace("\\", '/')
    # output_dir = str(output_dir).replace("\\", '/')

    hpc_s = f"ml.py --study-id {study_id} --output-dir {o_d.as_posix()} --dataset-folder {d_d.as_posix()} --n-job {n_job} --cv {cv} "
    for item in preprocessing_steps:
        hpc_s += f"--preprocessing-steps {item} "
    for item in class_healthy_label:
        hpc_s += f"--class-healthy-label {item} "
    for item in class_unhealthy_label:
        hpc_s += f"--class-unhealthy-label {item} "
    for item in meta_columns:
        hpc_s += f"--meta-columns {item} "
    for item in individual_to_ignore:
        hpc_s += f"--individual-to-ignore {item} "
    for item in classifiers:
        hpc_s += f"--classifiers {item} "

    if c is not None:
        hpc_s += f"--c {c} --gamma {gamma} "

    if pre_visu:
        hpc_s += "--pre_visu "

    if skip:
        hpc_s += "--skip"

    #print(hpc_s)
    with open("hpc_ln.txt", "a") as f:
        f.write(f"'{hpc_s}'" + "\n")
    with open("hpc.txt", "a") as f:
        f.write(f"'{hpc_s}'" + " ")


def main(
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    dataset_filepath: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    preprocessing_steps: List[str] = ["QN", "ANSCOMBE", "LOG", "STDS"],
    class_healthy_label: List[str] = [],
    class_unhealthy_label: List[str] = [],
    meta_columns: List[str] = [],
    add_feature: List[str] = [],
    n_scales: int = 8,
    sub_sample_scales: int = 1,
    n_splits: int = 5,
    n_repeats: int = 10,
    cv: str = "RepeatedKFold",
    classifiers: List[str] = [],
    wavelet_f0: int = 6,
    sfft_window: int = 60,
    dwt_window: str = "coif1",
    study_id: str = "study",
    sampling: str = "T",
    pre_visu: bool = True,
    output_qn_graph: bool = False,
    enable_qn_peak_filter: bool = False,
    enable_downsample_df: bool = False,
    n_job: int = 6,
    individual_to_ignore: List[str] = [],
    individual_to_keep: List[str] = [],
    individual_to_test: List[str] = [],
    save_model: bool = False,
    plot_2d_space: bool = False,
    export_fig_as_pdf: bool = False,
    skip: bool = False,
    export_hpc_string: bool = False,
    c: float = None,
    gamma: float = None
):
    """ML Main machine learning script\n
    Args:\n
        output_dir: Output directory.
        dataset_folder: Dataset input directory.
        preprocessing_steps: preprocessing steps.
        class_healthy_label: Label for healthy class.
        class_unhealthy_label: Label for unhealthy class.
        meta_columns: List of names of the metadata columns in the dataset file.plo
        meta_col_str: List of meta data to display on decision boundary plot.
        n_imputed_days: number of imputed days allowed in a sample.
        n_activity_days: number of activity days in a sample.
        n_scales: n scales in dyadic array [2^2....2^n].
        temp_file: csv file containing temperature features.
        hum_file: csv file containing humidity features.
        n_splits: Number of splits for repeatedkfold cv.
        n_repeats: Number of repeats for repeatedkfold cv.
        cv: RepeatedKFold|LeaveOneOut.
        wavelet_f0: Mother Wavelet frequency for CWT.
        sfft_window: STFT window size.
        farm_id: farm id.
        sampling: Activity bin resolution.
        pre_visu: Enable initial visualisation of dataset before classification.
        output_qn_graph: Output Quotient Normalisation steps figures.
        n_job: Number of threads to use for cross validation.
    """

    if export_hpc_string:
        build_hpc_string(
            study_id,
            output_dir,
            dataset_filepath,
            preprocessing_steps,
            class_healthy_label,
            class_unhealthy_label,
            classifiers,
            pre_visu,
            n_job,
            skip,
            meta_columns,
            cv,
            individual_to_ignore,
            c,
            gamma
        )
        return

    meta_columns = [x.replace("'", "") for x in meta_columns]
    preprocessing_steps = [x.replace("'", "") for x in preprocessing_steps]
    print(f"meta_columns={meta_columns}")
    print(f"preprocessing_steps={preprocessing_steps}")
    print(f"dataset_filepath={dataset_filepath}")
    print(f"output directory is {output_dir}")

    if skip:
        if output_dir.is_dir():
            print("output directory already exist. skip run.")
            return

    print(f"loading dataset file {dataset_filepath} ...")
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
        dataset_filepath,
        class_healthy_label,
        class_unhealthy_label,
        preprocessing_steps=preprocessing_steps,
        sampling=sampling,
        individual_to_keep=individual_to_keep,
        individual_to_ignore=individual_to_ignore
    )

    N_META = len(meta_columns)

    fig, ax = plt.subplots()
    sample = data_frame.iloc[0][:-N_META]
    ax.plot(sample)
    ax.set_ylabel('Activity Count', fontsize=18)
    ax.set_title('A Single Sample', fontsize=18)
    ax.set_xlabel('Time in seconds', fontsize=18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    filename = "sample.png"
    filepath = output_dir / filename
    print(filepath)
    fig.savefig(filepath)
    #plt.show()

    if pre_visu:
        # plotMeanGroups(n_scales, wavelet_f0, data_frame, label_series, N_META, output_dir + "/raw_before_qn/")
        #############################################################################################################
        #                                           VISUALISATION                                                   #
        #############################################################################################################
        animal_ids = data_frame["id"].astype(str).tolist()

        df_norm, _, time_freq_shape = apply_preprocessing_steps(
            meta_columns,
            sfft_window,
            dwt_window,
            wavelet_f0,
            animal_ids,
            data_frame.copy(),
            output_dir,
            ["QN"],
            class_healthy_label,
            class_unhealthy_label,
            clf_name="SVM_QN_VISU",
            n_scales=n_scales,
            keep_meta=True,
            output_qn_graph=output_qn_graph,
            sub_sample_scales=sub_sample_scales,
            enable_qn_peak_filter=enable_qn_peak_filter
        )

        plot_zeros_distrib(
            meta_columns,
            label_series,
            df_norm,
            output_dir,
            title="Percentage of zeros in activity per sample after normalisation",
        )
        plot_zeros_distrib(
            meta_columns,
            label_series,
            data_frame.copy(),
            output_dir,
            title="Percentage of zeros in activity per sample before normalisation",
        )
        # plot_mean_groups(
        #     sub_sample_scales,
        #     n_scales,
        #     sfft_window,
        #     wavelet_f0,
        #     dwt_window,
        #     df_norm,
        #     label_series,
        #     N_META,
        #     output_dir / "groups_after_qn",
        # )
        #
        # plot_mean_groups(
        #     sub_sample_scales,
        #     n_scales,
        #     sfft_window,
        #     wavelet_f0,
        #     dwt_window,
        #     data_frame,
        #     label_series,
        #     N_META,
        #     output_dir / "groups_before_qn",
        # )

        plot_umap(
            meta_columns,
            data_frame.copy(),
            output_dir / "umap",
            label_series,
            title="UMAP time domain before normalisation",
        )

        plot_umap(
            meta_columns,
            df_norm.copy(),
            output_dir / "umap",
            label_series,
            title="UMAP time domain after normalisation",
        )

        plot_time_pca(
            N_META,
            data_frame.copy(),
            output_dir / "pca",
            label_series,
            title="PCA time domain before normalisation",
        )
        plot_time_pca(
            N_META,
            df_norm,
            output_dir / "pca",
            label_series,
            title="PCA time domain after normalisation",
        )

        plot_time_pls(
            N_META,
            data_frame.copy(),
            output_dir / "pls",
            label_series,
            title="PLS time domain before normalisation",
        )
        plot_time_pls(
            N_META,
            df_norm,
            output_dir / "pls",
            label_series,
            title="PLS time domain after normalisation",
        )

        plot_time_lda(
            N_META,
            data_frame.copy(),
            output_dir / "lda",
            label_series,
            title="LDA time domain before normalisation",
        )
        plot_time_lda(
            N_META,
            data_frame.copy(),
            output_dir / "lda",
            label_series,
            title="LDA time domain after normalisation",
        )

        ntraces = 2
        idx_healthy, idx_unhealthy = plot_groups(
            N_META,
            animal_ids,
            class_healthy_label,
            class_unhealthy_label,
            output_dir,
            data_frame.copy(),
            title="Raw data",
            xlabel="Time",
            ylabel="activity",
            ntraces=ntraces,
        )
        plot_groups(
            N_META,
            animal_ids,
            class_healthy_label,
            class_unhealthy_label,
            output_dir,
            df_norm,
            title="Normalised(Quotient Norm) samples",
            xlabel="Time",
            ylabel="activity",
            idx_healthy=idx_healthy,
            idx_unhealthy=idx_unhealthy,
            stepid=2,
            ntraces=ntraces,
        )

        plot_crepuscular(output_dir, data_frame.copy(), filename="raw_median_peak", ylabel="Activity count")
        plot_crepuscular(output_dir, df_norm.copy(), filename="qn_median_peak", ylabel="Activity count")
        ################################################################################################################

    step_slug = "_".join(preprocessing_steps)
    steps_ = []
    if "APPEND" in preprocessing_steps:
        steps_.append(preprocessing_steps[0 : preprocessing_steps.index("APPEND")])
        steps_.append(
            preprocessing_steps[preprocessing_steps.index("APPEND") + 1 :]
        )
    else:
        steps_ = [preprocessing_steps]

    sample_dates = pd.to_datetime(
        data_frame["date"], format="%d/%m/%Y"
    ).values.astype(np.float16)
    animal_ids = data_frame["id"].astype(str).tolist()
    df_processed_list = []
    for preprocessing_steps in steps_:
        (
            df_processed,
            df_processed_meta,
            time_freq_shape,
        ) = apply_preprocessing_steps(
            meta_columns,
            sfft_window,
            dwt_window,
            wavelet_f0,
            animal_ids,
            data_frame.copy(),
            output_dir,
            preprocessing_steps,
            class_healthy_label,
            class_unhealthy_label,
            clf_name="SVM",
            n_scales=n_scales,
            sub_sample_scales=sub_sample_scales,
            enable_qn_peak_filter=enable_qn_peak_filter,
        )
        df_processed_list.append(df_processed)

    df = pd.concat(df_processed_list, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    target = df.pop("target")
    health = df.pop("health")
    df_processed = pd.concat([df, target, health], 1)

    plot_umap(
        meta_columns,
        df_processed_meta.copy(),
        output_dir / f"umap_{step_slug}",
        label_series,
        title=f"UMAP after {step_slug}",
    )

    plot_time_pca(
        N_META,
        df_processed_meta,
        output_dir / f"pca_{step_slug}",
        label_series,
        title=f"PCA after {step_slug}",
    )

    plot_time_pls(
        N_META,
        df_processed_meta.copy(),
        output_dir / f"pls_{step_slug}",
        label_series,
        title=f"PLS after {step_slug}",
    )

    process_ml(
        classifiers,
        add_feature,
        df_meta,
        output_dir,
        animal_ids,
        sample_dates,
        df_processed,
        study_id,
        step_slug,
        n_splits,
        n_repeats,
        sampling,
        enable_downsample_df,
        label_series,
        class_healthy_label,
        class_unhealthy_label,
        meta_columns,
        save_model=save_model,
        cv=cv,
        n_job=n_job,
        individual_to_test=individual_to_test,
        plot_2d_space=plot_2d_space,
        export_fig_as_pdf=export_fig_as_pdf,
        C=c,
        gamma=gamma
    )


if __name__ == "__main__":
    typer.run(main)
