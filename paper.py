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

import typer
import build_dataset
import run_ml
from bootstrap import boot_roc_curve
from pathlib import Path


def main(
    data_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    create_dataset: bool = True,
    n_bootstrap: int = 100,
    n_job: int = 6,
):
    """Script to reproduce paper results\n
    Args:\n
        data_dir: Directory containing the Cats data .csv.
    """
    out_dir = data_dir / "output_test6"

    if create_dataset:
        build_dataset.run(
            w_size=[10, 30, 60, 120],
            threshs=[10, 20],
            n_peaks=[1],
            data_dir=data_dir,
            out_dir=out_dir,
            n_job=n_job,
        )

    datasets = [x for x in Path(out_dir).glob("**/*/samples.csv")]

    assert len(datasets) > 0, f"There is no dataset in {out_dir}."

    results = []
    for dataset in datasets:
        run_ml.run(
            dataset_filepath=dataset,
            out_dir=out_dir,
            preprocessing_steps=[],
            n_job=n_job,
        )
        res = boot_roc_curve.main(
            dataset.parent.parent, n_bootstrap=n_bootstrap, n_job=n_job
        )
        results.append(res)
    boot_roc_curve.boostrap_auc_peak(results, out_dir)


if __name__ == "__main__":
    data_dir = Path("E:/Cats")
    main(data_dir)
    # typer.run(main)
