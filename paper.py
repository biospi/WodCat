#
# Author: Axel Montout <axel.montout <a.t> bristol.ac.uk>
#
# Copyright (C) 2020  Biospi Laboratory for Medical Bioinformatics, University of Bristol, UK
#
# This file is part of WodCat.
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
import numpy as np
import pandas as pd
import typer
import matplotlib

matplotlib.use("Agg")
import build_dataset
import run_ml
import boot_roc_curve
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utils.utils import purge_hpc_file, create_batch_script
import plotly.graph_objects as go
import scipy
import os


def find_samples_csv(root_dir):
    result = []
    folders = [f for f in root_dir.iterdir() if f.is_dir()]
    for f in folders:
        sample_dir = f / "dataset" / "samples.csv"
        result.append(sample_dir)

    return sorted(result)


def main(
    data_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    create_dataset: bool = False,
    export_hpc_string: bool = False,
    bc_username: str = "sscm012844",
    uob_username: str = "fo18103",
    out_dirname: str = "paper_allclf",
    dataset_path: Path = Path("dataset.csv"),
    n_bootstrap: int = 100,
    ml_exist: bool = False,
    skip_ml: bool = False,
    regularisation: bool = False,
    n_job: int = 30,
    build_heatmap: bool = False,
):
    """Script to reproduce paper results\n
    Args:\n
        data_dir: Directory containing the Cats data .csv.
        export_hpc_string: Create .sh submission file for Blue Crystal/Blue Pebble. Please ignore if running locally.
    """
    out_dir = data_dir / out_dirname

    # build_dataset.run(
    #     w_size=[15],
    #     threshs=[10],
    #     n_peaks=[0],
    #     data_dir=data_dir,
    #     out_dir=out_dir,
    #     max_sample=100,
    #     day_windows=["All"],
    #     n_job=n_job,
    #     dataset_path=dataset_path,
    #     use_age_as_feature=True
    # )
    # exit()

    if build_heatmap:
        # plot all data heatmap
        build_dataset.run(
            w_size=[15],
            threshs=[10],
            n_peaks=[1],
            data_dir=data_dir,
            out_dir=out_dir,
            max_sample=100,
            day_windows=["All"],
            n_job=n_job,
            dataset_path=dataset_path,
            out_heatmap=True,
            bin="T",
        )
        exit()

    if create_dataset:
        if not out_dir.exists():
            for max_sample in [150]:
                build_dataset.run(
                    w_size=[30],
                    threshs=[30],
                    n_peaks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                    data_dir=data_dir,
                    out_dir=out_dir,
                    max_sample=max_sample,
                    day_windows=["All"],
                    n_job=n_job,
                    dataset_path=dataset_path,
                )

    print("searching dataset...")
    # datasets = sorted([x for x in Path(out_dir).glob("**/*/samples.csv")])
    ##datasets = sorted(Path(out_dir).rglob("samples.csv"))
    datasets = find_samples_csv(out_dir)

    #print(f"datasets={datasets}")
    # meta_columns = sorted([pd.read_csv(x).values.flatten().tolist() for x in Path(out_dir).glob("**/*/meta_columns.csv")])
    # print(f"meta_columns={meta_columns}")

    assert (
        len(datasets) > 0
    ), f"There is no dataset in {out_dir}. create_dataset={create_dataset}"

    if export_hpc_string:  # ignore this if you do not use Blue Crystal(UoB)
        purge_hpc_file("hpc.txt")
        purge_hpc_file("hpc_ln.txt")

    results = []
    for i, dataset in enumerate(datasets):
        # if int(dataset.parent.parent.name.split('_')[-1]) < 4: #todo remove
        #     continue
        n_peak = int(dataset.parent.parent.stem.split("_")[-1])
        meta_columns_file = dataset.parent / "meta_columns.csv"
        meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
        print(f"dataset={dataset}")
        # print(f"meta_columns={meta_columns}")
        if ml_exist:  # if you already ran the classification pipeline on hpc
            print("Parsing existing results...")
            ml_out = [x.parent for x in dataset.parent.parent.glob("**/fold_data")]
            print(f"ml_out={ml_out}")
            for out_ml_dir in ml_out:
                print(f"out_ml_dir={out_ml_dir}")
                res = boot_roc_curve.main(
                    out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
                )
                results.append(res)
        else:
            print("Running machine learning pipeline...")
            if n_peak == 0:  # process dataset wit age as the only feature
                out_ml_dir, status = run_ml.run(
                    preprocessing_steps=[""],
                    export_hpc_string=export_hpc_string,
                    regularisation=regularisation,
                    meta_columns=meta_columns,
                    dataset_filepath=dataset,
                    out_dir=out_dir,
                    skip=skip_ml,
                    n_job=n_job,
                    pre_visu=False,
                    n_peak=n_peak,
                )
                res = boot_roc_curve.main(
                    out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
                )
                results.append(res)
            else:
                for clf in ["lreg", "rbf", "knn", "dtree"]:
                    for preprocessing_steps in [
                        [""],
                        ["L1"],
                        ["L1", "L1SCALE", "ANSCOMBE"],
                    ]:
                        pre_visu = False
                        out_ml_dir, status = run_ml.run(
                            preprocessing_steps=preprocessing_steps,
                            export_hpc_string=export_hpc_string,
                            regularisation=regularisation,
                            meta_columns=meta_columns,
                            dataset_filepath=dataset,
                            out_dir=out_dir,
                            skip=skip_ml,
                            n_job=n_job,
                            pre_visu=pre_visu,
                            n_peak=n_peak,
                            clf=clf
                        )

                        if export_hpc_string:
                            continue

                        res = boot_roc_curve.main(
                            out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
                        )
                        results.append(res)

    # create submission file for Blue Crystal(UoB), please ignore if running on local computer
    if export_hpc_string:
        with open("hpc_ln.txt") as file:
            command_list = [line.rstrip() for line in file]
        create_batch_script(uob_username, bc_username, command_list, len(command_list))
        return

    print("Create n peak comparison ROC curve...")
    boot_roc_curve.boostrap_auc_peak(results, out_dir)
    #boot_roc_curve.boostrap_auc_peak_delta(results, out_dir)

    #print("Create boxplot best model")
    #best_model_boxplot(results, out_dir)


def best_model_boxplot(results, out_dir):
    results.sort(key=lambda x: x[12])
    best_model = np.array(results[-1][14])

    aucs = [np.array(x[14]) - best_model for x in results]
    labels = [f"{r[6]}_{r[16][0].parent.parent.stem}" for r in results]

    # format labels to human readable
    labels_formatted = []
    for l in labels:
        if l == "0__LeaveOneOut":
            labels_formatted.append("Age")
        if l == "1_L1_L1SCALE_ANSCOMBE_LeaveOneOut":
            labels_formatted.append("Activity 1 peak")
        if l == "22_L1_L1SCALE_ANSCOMBE_LeaveOneOut":
            labels_formatted.append("Activity 22 peaks")

    fig = go.Figure()
    for auc, label in zip(aucs, labels_formatted):
        if np.sum(auc) == 0:
            p_value = np.nan
        else:
            p_value = scipy.stats.wilcoxon(auc, alternative="less").pvalue
        # print(p_value)

        label_with_p_value = (
            f"{label} (p-value: {p_value:.2e})"
            if not np.isnan(p_value)
            else f"{label} (p-value: NaN)"
        )
        fig.add_trace(go.Box(y=auc, name=label))

    fig.update_layout(
        title="Best Model AUC Comparison",
        xaxis_title="Model",
        yaxis_title="AUC(Delta)",
        xaxis={"tickangle": 45},  # Rotate labels for better readability
        showlegend=True,
        font=dict(family="Times New Roman", size=12, color="black"),
    )
    filepath = str(out_dir / "best_vs_others_nop.html")
    print(filepath)
    fig.write_html(filepath)

    # # Set size and DPI for the PNG export
    # width_in_inches = 1
    # height_in_inches = 1.5
    # dpi = 500
    #
    # # Convert inches to pixels
    # width_in_pixels = 715
    # height_in_pixels = 930
    #
    # # Define the file path for the PNG
    # png_filepath = str(out_dir / 'best_vs_others.png')
    #
    # # Export as PNG
    # fig.write_image(png_filepath, width=width_in_pixels, height=height_in_pixels, scale=1)
    #
    # print(f"Saved to {png_filepath}")


if __name__ == "__main__":
    # main(data_dir=Path("E:/Cats"),
    #      dataset_path=Path('E:/dataset.csv'),
    #      out_dirname="paper_debug_regularisation_36",
    #      create_dataset=False,
    #      ml_exist=True)
    typer.run(main)
