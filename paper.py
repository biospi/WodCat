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
matplotlib.use('Agg')
import build_dataset
import run_ml
import boot_roc_curve
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utils.utils import purge_hpc_file, create_batch_script


def main(
    data_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    create_dataset: bool = True,
    export_hpc_string: bool = False,
    bc_username: str = 'sscm012844',
    uob_username: str = 'fo18103',
    out_dirname: str = 'paper',
    clf: str = 'rbf',
    dataset_path: Path = Path("dataset.csv"),
    n_bootstrap: int = 1000,
    ml_exist: bool = False,
    skip_ml: bool = False,
    regularisation: bool = False,
    n_job: int = 28,
    build_heatmap: bool = False
):
    """Script to reproduce paper results\n
    Args:\n
        data_dir: Directory containing the Cats data .csv.
        export_hpc_string: Create .sh submission file for Blue Crystal/Blue Pebble. Please ignore if running locally.
    """
    out_dir = data_dir / out_dirname

    if build_heatmap:
        #plot all data heatmap
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
            bin='T'
        )
        exit()

    if create_dataset:
        for max_sample in [100]:
            build_dataset.run(
                w_size=[15],
                threshs=[10],
                n_peaks=[1, 2],
                data_dir=data_dir,
                out_dir=out_dir,
                max_sample=max_sample,
                day_windows=["All"],
                n_job=n_job,
                dataset_path=dataset_path
            )

    datasets = sorted([x for x in Path(out_dir).glob("**/*/samples.csv")])

    print(f"datasets={datasets}")
    # meta_columns = sorted([pd.read_csv(x).values.flatten().tolist() for x in Path(out_dir).glob("**/*/meta_columns.csv")])
    # print(f"meta_columns={meta_columns}")

    assert len(datasets) > 0, f"There is no dataset in {out_dir}. create_dataset={create_dataset}"

    if export_hpc_string: #ignore this if you do not use Blue Crystal(UoB)
        purge_hpc_file("hpc.txt")
        purge_hpc_file("hpc_ln.txt")

    results = []
    for i, dataset in enumerate(datasets):
        # if int(dataset.parent.parent.name.split('_')[-1]) < 4: #todo remove
        #     continue
        n_peak = int(dataset.parent.parent.stem.split('_')[-1])
        meta_columns_file = dataset.parent / "meta_columns.csv"
        meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
        print(f"dataset={dataset}")
        print(f"meta_columns={meta_columns}")
        if ml_exist: #if you already ran the classification pipeline on hpc
            print("Parsing existing results...")
            ml_out = [x.parent for x in dataset.parent.parent.glob("**/fold_data")]
            for out_ml_dir in ml_out:
                print(f"out_ml_dir={out_ml_dir}")
                res = boot_roc_curve.main(
                    out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
                )
                results.append(res)
        else:
            print("Running machine learning pipeline...")
            for preprocessing_steps in [
                [""],
                ["L1"],
                ["L1", "L1SCALE", "ANSCOMBE"],
                ["L1", "L1SCALE", "ANSCOMBE", "LOG"]
            ]:
                pre_visu = True #export grapth just for the first run to save storage space
                if i == 0:
                    pre_visu = True

                out_ml_dir, status = run_ml.run(
                    preprocessing_steps=preprocessing_steps,
                    export_hpc_string=export_hpc_string,
                    regularisation=regularisation,
                    meta_columns=meta_columns,
                    dataset_filepath=dataset,
                    out_dir=out_dir,
                    skip=skip_ml,
                    n_job=n_job,
                    clf=clf,
                    pre_visu=pre_visu,
                    n_peak=n_peak
                )
                if export_hpc_string:
                    continue
                res = boot_roc_curve.main(
                    out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
                )
                results.append(res)

    #create submission file for Blue Crystal(UoB), please ignore if running on local computer
    if export_hpc_string:
        with open('hpc_ln.txt') as file:
            command_list = [line.rstrip() for line in file]
        create_batch_script(uob_username, bc_username, command_list, len(command_list))
        return

    print("Create n peak comparison ROC curve...")
    boot_roc_curve.boostrap_auc_peak(results, out_dir)

    print("Create boxplot best model")
    best_model_boxplot(results, out_dir)


def best_model_boxplot(results, out_dir):
    results.sort(key=lambda x: x[12])
    best_model = np.array(results[-1][14])

    aucs = [np.array(x[14]) - best_model for x in results]
    labels = []
    for r in results:
        l = f"{r[6]}_{r[16][0].parent.parent.stem}"
        labels.append(l)

    fig, ax = plt.subplots(figsize=(len(results)*1.1, 6))
    ax.boxplot(aucs, labels=labels)
    plt.xticks(rotation=45)
    ax.set_title('Best model AUC')
    ax.set_ylabel('AUC Values')
    ax.grid()
    plt.tight_layout()
    fig.savefig(out_dir / 'box_plot.png')


if __name__ == "__main__":
    #data_dir = Path("/mnt/storage/scratch/axel/cats")
    typer.run(main)
