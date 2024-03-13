import typer
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

from model.data_loader import load_activity_data
from preprocessing.preprocessing import apply_preprocessing_steps

IDS = [
    "'Greg'",
    "'Henry'",
    "'Tilly'",
    # "'Maisie'",
    "'Sookie'",
    # "'Oliver_F'",
    "'Ra'",
    "'Hector'",
    "'Jimmy'",
    # "'MrDudley'",
    "'Kira'",
    # "'Lucy'",
    "'Louis'",
    "'Luna_M'",
    "'Wookey'",
    "'Logan'",
    "'Ruby'",
    "'Kobe'",
    "'Saffy_J'",
    "'Enzo'",
    "'Milo'",
    "'Luna_F'",
    "'Oscar'",
    "'Kia'",
    "'Cat'",
    "'AlfieTickles'",
    "'Phoebe'",
    "'Harvey'",
    "'Mia'",
    "'Amadeus'",
    "'Marley'",
    "'Loulou'",
    "'Bumble'",
    "'Skittle'",
    "'Charlie_O'",
    "'Ginger'",
    "'Hugo_M'",
    "'Flip'",
    "'Guinness'",
    "'Chloe'",
    "'Bobby'",
    "'QueenPurr'",
    "'Jinx'",
    "'Charlie_B'",
    "'Thomas'",
    "'Sam'",
    "'Max'",
    "'Oliver_S'",
    "'Millie'",
    "'Clover'",
    "'Bobbie'",
    "'Gregory'",
    "'Kiki'",
    "'Hugo_R'",
    "'Shadow'",
]

IDS2 = [
    "Mia",
    "Loulou",
    "Sam",
    "Enzo",
    "Amadeus",
    "Bobbie",
    "Kobe",
    "Hugo_R",
    "Wookey",
    "Millie",
]


def main(
    dataset_file: Path = typer.Option(
        ..., exists=False, file_okay=True, dir_okay=False, resolve_path=True
    ),
    output_dir: Path = typer.Option(
        ..., exists=False, file_okay=False, dir_okay=True, resolve_path=True
    ),
    ids: List[str] = [],
    ids_g1: List[str] = [],
    ids_g2: List[str] = [],
    meta_columns: List[str] = [],
    preprocessing_steps: List[str] = ["L1"],
    class_healthy_label: List[str] = ["0.0"],
    class_unhealthy_label: List[str] = ["1.0"],
    individual_to_ignore: List[str] = ["MrDudley", "Oliver_F", "Lucy"],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"ids={ids}")
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
        preprocessing_steps=preprocessing_steps,
        individual_to_ignore=individual_to_ignore
    )

    data_frame_time, _, _ = apply_preprocessing_steps(
        meta_columns,
        None,
        None,
        None,
        None,
        data_frame.copy(),
        output_dir,
        ["L1"],
        class_healthy_label,
        class_unhealthy_label,
        clf_name="SVM_QN_VISU",
        keep_meta=True,
        output_qn_graph=False
    )

    print(data_frame_time)

    for id in ids:
        id = id.strip().replace("'", "")
        A = data_frame_time[data_frame_time["name"] == id]
        make_plot(A, output_dir, id, len(meta_columns))


def make_plot(A, output_dir, id, n_meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    # id='_'.join(id).replace("'",'')
    health = A["health"].mean()
    df_activity = A.iloc[:, :-n_meta]
    print(df_activity)
    title = f"Samples health={health}"
    plt.clf()
    fig = df_activity.T.plot(
        kind="line",
        subplots=False,
        grid=True,
        legend=False,
        title=title,
        alpha=0.7,
        xlabel="Time(s)",
        ylabel="Activity count",
    ).get_figure()
    #plt.ylim(0, 70)
    plt.xticks(rotation=45)
    plt.tight_layout()
    filepath = output_dir / f"{id}.png"
    print(filepath)
    fig.savefig(filepath)


if __name__ == "__main__":
    # dataset = Path(
    #         "E:/Cats/paper_debug_regularisation_8/All_100_10_060_001/dataset/samples.csv"
    #     )
    # meta_columns_file = dataset.parent / "meta_columns.csv"
    # meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()
    #
    # main(
    #     dataset_file=dataset,
    #     meta_columns=meta_columns,
    #     output_dir=Path("E:/Cats/paper_visu/All_100_10_060_001"),
    #     ids=IDS2,
    # )

    dataset = Path(
            "E:/Cats/paper_debug_regularisation_8/All_100_10_060_003/dataset/samples.csv"
        )
    meta_columns_file = dataset.parent / "meta_columns.csv"
    meta_columns = pd.read_csv(meta_columns_file).values.flatten().tolist()

    main(
        dataset_file=dataset,
        meta_columns=meta_columns,
        output_dir=Path("E:/Cats/paper_visu/All_100_10_060_003"),
        ids=IDS2,
    )
