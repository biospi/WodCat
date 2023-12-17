from pathlib import Path
from typing import List

import typer

from pipeline import ml


def run(
    dataset_filepath: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    out_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    clf: str = "rbf",
    preprocessing_steps: List[str] = ["QN"],
    meta_columns: List[str] = [],
    cv: str = "LeaveOneOut",
    export_hpc_string: bool = False,
    n_job: int = 4,
):
    """Thesis script runs the cats study
    Args:\n
        out_parent: Output directory
        dataset_parent: Dataset directory
    """

    slug = "_".join(preprocessing_steps)
    output_dir = out_dir / dataset_filepath.parent.parent.stem / clf / f"{slug}_{cv}"
    ml.main(
        output_dir=output_dir,
        dataset_filepath=dataset_filepath,
        preprocessing_steps=preprocessing_steps,
        meta_columns=meta_columns,
        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        individual_to_keep=[],
        classifiers=[clf],
        class_healthy_label=["0.0"],
        class_unhealthy_label=["1.0"],
        n_scales=8,
        n_splits=5,
        n_repeats=10,
        n_job=n_job,
        study_id="cat",
        cv=cv,
        output_qn_graph=False,
        pre_visu=False,
        plot_2d_space=False,
        export_hpc_string=export_hpc_string
    )
    return output_dir


if __name__ == "__main__":
    typer.run(run)
