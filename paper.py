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
import pandas as pd
import typer
import build_dataset
import run_ml
from bootstrap import boot_roc_curve
from pathlib import Path

from utils.utils import purge_hpc_file, create_batch_script


def main(
    data_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    create_dataset: bool = True,
    export_hpc_string: bool = True,
    bc_username: str = 'sscm012844',
    n_bootstrap: int = 1000,
    n_job: int = 6,
):
    """Script to reproduce paper results\n
    Args:\n
        data_dir: Directory containing the Cats data .csv.
        export_hpc_string: Create .sh submission file for Blue Crystal/Blue Pebble. Please ignore if running locally.
    """
    out_dir = data_dir / "data4"

    if create_dataset:
        build_dataset.run(
            w_size=[10],
            threshs=[10],
            n_peaks=[1, 2],
            data_dir=data_dir,
            out_dir=out_dir,
            n_job=n_job,
        )

    datasets = [x for x in Path(out_dir).glob("**/*/samples.csv")]
    meta_columns = [pd.read_csv(x).values.flatten().tolist() for x in Path(out_dir).glob("**/*/meta_columns.csv")]

    assert len(datasets) > 0, f"There is no dataset in {out_dir}."

    if export_hpc_string: #ignore this if you do not use Blue Crystal(UoB)
        purge_hpc_file("hpc.txt")
        purge_hpc_file("hpc_ln.txt")

    results = []
    for meta_columns, dataset in zip(meta_columns, datasets):
        for preprocessing_steps in [[],
                                    ["QN"],
                                    ["STDS"],
                                    ["QN", "ANSCOMBE", "LOG"],
                                    ["QN", "ANSCOMBE", "LOG", "STDS"]
                                    ]:
            run_ml.run(
                preprocessing_steps=preprocessing_steps,
                export_hpc_string=export_hpc_string,
                meta_columns=meta_columns,
                dataset_filepath=dataset,
                out_dir=out_dir,
                n_job=n_job,
            )
            if export_hpc_string:
                continue

            res = boot_roc_curve.main(
                dataset.parent.parent, n_bootstrap=n_bootstrap, n_job=n_job
            )
            results.append(res)

    #create submission file for Blue Crystal(UoB), please ignore if running on local computer
    if export_hpc_string:
        with open('hpc_ln.txt') as file:
            command_list = [line.rstrip() for line in file]
        create_batch_script(bc_username, command_list, len(command_list))
        return

    boot_roc_curve.boostrap_auc_peak(results, out_dir)


if __name__ == "__main__":
    data_dir = Path("E:/Cats")
    main(data_dir)
    #typer.run(main)
