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
    ml.main(
        output_dir=out_dir / dataset_filepath.parent.parent.stem / clf / f"{slug}_{cv}",
        dataset_filepath=dataset_filepath,
        preprocessing_steps=preprocessing_steps,
        meta_columns=[
            "label",
            "id",
            "imputed_days",
            "date",
            "health",
            "target",
            "age",
            "name",
            "mobility_score",
            "max_sample",
            "n_peak",
            "w_size",
            "n_top"
        ],
        meta_col_str=[],
        individual_to_ignore=["MrDudley", "Oliver_F", "Lucy"],
        individual_to_keep=[],
        classifiers=[clf],
        n_imputed_days=-1,
        n_activity_days=-1,
        class_healthy_label=["0.0"],
        class_unhealthy_label=["1.0"],
        n_scales=8,
        n_splits=5,
        n_repeats=10,
        n_job=n_job,
        study_id="cat",
        cv=cv,
        output_qn_graph=False,
        enable_qn_peak_filter=False,
        plot_2d_space=False,
        export_hpc_string=export_hpc_string
    )


if __name__ == "__main__":
    typer.run(run)
