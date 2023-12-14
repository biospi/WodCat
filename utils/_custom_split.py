import random
import warnings

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

from sklearn.utils import resample


import plotly.graph_objs as go
from plotly.subplots import make_subplots

np.random.seed(0)


def plot_heatmap(
    X,
    timestamps,
    animal_ids,
    out_dir,
    title="Heatmap",
    filename="heatmap.html",
    yaxis="Count",
    xaxis="Time bin(1D)",
):
    fig = make_subplots(rows=1, cols=1)
    trace = go.Heatmap(
        z=X.T,
        x=timestamps,
        y=animal_ids,
        colorscale="Viridis",
    )
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)
    # fig.show()
    # create_rec_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    return trace, title


def remove_testing_samples(activity_file, out_dir, ifold, df, to_remove, nmin=10080):
    print("removing testing samples from dataset....")
    df_without_test = df.copy()
    df_without_test["date_datetime"] = pd.to_datetime(df_without_test["date_str"])
    for test_sample in to_remove:
        animal_id = int(test_sample[0])
        date = pd.to_datetime(test_sample[2], format="%d/%m/%Y")
        date_range = pd.date_range(end=date, periods=nmin, freq="T")
        # grab test sample from full dataset and remove activity values
        df_without_test.loc[
            df_without_test[[str(animal_id), "date_datetime"]]["date_datetime"].isin(
                date_range
            ),
            str(animal_id),
        ] = -1
    df_resampled = df_without_test.resample("D", on="date_datetime").sum()
    filtered_columns = [x for x in df_resampled.columns if x.isdigit()]
    df_resampled = df_resampled[filtered_columns]
    df_resampled[df_resampled._get_numeric_data() < 0] = np.nan
    timestamps = [pd.to_datetime(x) for x in df_resampled.index.values]
    plot_heatmap(
        df_resampled.values,
        timestamps,
        [f"_{x}" for x in filtered_columns],
        title=f"Fold {ifold} training data with testing samples removed",
        out_dir=out_dir / Path("CV"),
        filename=f"fold_{ifold}.html",
    )
    print("removed testing data. ready for imputation.")
    filename = f"{activity_file.stem}_{ifold}.csv"
    out = out_dir / Path("CV") / filename
    print(out)
    df_without_test[df_without_test._get_numeric_data() < 0] = np.nan
    df_without_test.to_csv(out)
    return df_without_test, out, filtered_columns


class LeaveNOut:
    def __init__(
        self,
        animal_ids,
        sample_idx,
        max_comb=-1,
        leaven=2,
        n_test_samples_th=-1,
        stratified=False,
        verbose=True,
        individual_to_test=None
    ):
        self.nfold = 0
        self.leaven = leaven
        self.max_comb = max_comb
        self.verbose = verbose
        self.stratified = stratified
        self.n_test_samples_th = n_test_samples_th
        self.sample_idx = np.array(sample_idx).flatten()
        self.animal_ids = np.array(animal_ids).flatten()
        self.info_list = []
        self.individual_to_test = individual_to_test

    def get_n_splits(self):
        return self.nfold

    def get_fold_info(self, i):
        return self.info_list[i]

    def split(self, X, y, group=None):
        info_list = []
        df = pd.DataFrame(
            np.hstack(
                (
                    y.reshape(y.size, 1),
                    self.animal_ids.reshape(self.animal_ids.size, 1),
                    self.sample_idx.reshape(self.sample_idx.size, 1),
                )
            )
        )
        # df.to_csv("F:/Data2/test.csv")
        # df = pd.read_csv("F:/Data2/test.csv", index_col=False)
        df = df.apply(pd.to_numeric, downcast="integer")
        df.columns = ["target", "animal_id", "sample_idx"]
        # if self.individual_to_test is not None and len(self.individual_to_test) > 0:
        #     df = df.loc[df['animal_id'].isin(self.individual_to_test)]
        ##df.index = df["sample_idx"]

        groupby_target = pd.DataFrame(df.groupby("animal_id")["target"].apply(list))
        groupby_target["animal_id"] = groupby_target.index
        groupby_sample = pd.DataFrame(df.groupby("animal_id")["sample_idx"].apply(list))
        groupby_sample["animal_id"] = groupby_sample.index

        df_ = groupby_target.copy()
        df_["sample_idx"] = groupby_sample["sample_idx"]
        df_ = df_[["animal_id", "target", "sample_idx"]]

        if self.verbose:
            print("DATASET:")
            print(df_)

        a = df_["animal_id"].tolist()
        comb = []
        cpt = 0
        for j, subset in enumerate(itertools.combinations(a, self.leaven)):
            if len(subset) != self.leaven:
                continue
            if subset not in comb or subset[::-1] not in comb:
                comb.append(subset)
                cpt += 1
            # if cpt > self.max_comb:
            #     break
        comb = np.array(comb)

        if self.max_comb > 0:
            df_com = pd.DataFrame(comb)
            comb = df_com.sample(self.max_comb).values

        training_idx = []
        testing_idx = []
        len_check = []
        map = dict(df["sample_idx"])
        map = dict(zip(map.values(), map.keys()))
        for i, c in enumerate(comb):
            test_idx = df_[df_["animal_id"].isin(c)]["sample_idx"].tolist()
            all_test_idx = sum(test_idx, [])
            all_test_idx = [map[x] for x in all_test_idx]
            train_idx = df_[~df_["animal_id"].isin(c)]["sample_idx"].tolist()
            all_train_idx = sum(train_idx, [])
            all_train_idx = [map[x] for x in all_train_idx]

            if self.stratified:
                temp = []
                for e in test_idx:
                    temp.append(df[df["sample_idx"].isin(e)]["target"].tolist())

                s1 = np.unique(np.array(temp[0]))

                if len(temp) == self.leaven:
                    s2 = np.unique(np.array(temp[1]))
                    if s1.size != 1 and s2.size != 1:
                        # samples for the 2 left out animals are not the same target
                        continue
                    s = np.array([s1[0], s2[0]])
                    if (
                        np.unique(s).size != self.leaven
                    ):  # need 1 healthy and 1 unhealthy
                        continue
                else:
                    if s1.size != 1:
                        continue
                    s = np.array(s1[0])
                    if np.unique(s).size != 1:
                        continue
            if np.unique(y[all_train_idx]).size == 1:
                warnings.warn(
                    "Cannot use fold for training! Only 1 target in FOLD %d" % i
                )
                continue
            # if len(all_test_idx) < self.n_test_samples_th:
            #     continue
            if self.individual_to_test is not None and len(self.individual_to_test) > 0:
                a = int(float(np.unique(self.animal_ids[all_test_idx]).tolist()[0]))
                b = np.array(self.individual_to_test).astype(int)
                print(a, b)
                if a not in b:
                    continue

            training_idx.append(all_train_idx)
            testing_idx.append(all_test_idx)
            len_check.append(len(test_idx))
            if self.verbose:
                info = {
                    "FOLD": i,
                    "SAMPLE TRAIN IDX": all_train_idx,
                    "SAMPLE TEST IDX": all_test_idx,
                    "TEST TARGET": np.unique(y[all_test_idx]).tolist(),
                    "TRAIN TARGET": np.unique(y[all_train_idx]).tolist(),
                    "TEST ANIMAL ID": np.unique(self.animal_ids[all_test_idx]).tolist(),
                    "TRAIN ANIMAL ID": np.unique(self.animal_ids[all_train_idx]).tolist(),
                }
                self.info_list.append(info)
                #print(info)

        len_check = np.array(len_check)
        if len_check[len_check > self.leaven].size > 0:
            raise ValueError("fold contains more than 2 testing sample!")

        self.nfold = len(training_idx)
        print(
            "LeaveNOut could build %d unique folds. stratification=%s"
            % (self.nfold, self.stratified)
        )
        for n in range(len(training_idx)):
            yield np.array(training_idx[n]), np.array(testing_idx[n])


class BootstrapCustom_:
    def __init__(
        self,
        animal_ids,
        n_iterations = 100,
        random_state=0,
        n_bootstraps=3,
        stratified=False,
        verbose=False,
        individual_to_test=None
    ):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_bootstraps = n_bootstraps
        self.nfold = 0
        self.verbose = verbose
        self.stratified = stratified
        self.animal_ids = np.array(animal_ids).flatten()
        self.info_list = []

    def get_n_splits(self):
        return self.nfold

    def get_fold_info(self, i):
        return self.info_list[i]

    def split(self, X, y, group=None):

        train_list = []
        test_list = []
        for i in range(self.n_iterations):
            n_size = len(np.unique(self.animal_ids)) - 1
            train = resample(np.unique(self.animal_ids), n_samples=n_size, random_state=self.random_state)  # Sampling with replacement..whichever is not used in training data will be used in test data
            train_list.append(train)
            test = np.array([x for x in np.unique(self.animal_ids) if
                             x.tolist() not in train.tolist()])  # picking rest of the data not considered in training sample
            test = [random.choice(test)]
            test_list.append(test)

        df = pd.DataFrame(
            np.hstack(
                (
                    y.reshape(y.size, 1),
                    self.animal_ids.reshape(self.animal_ids.size, 1)
                )
            )
        )
        df.columns = ["target", "animal_id"]
        df["sample_idx"] = df.index

        training_idx = []
        testing_idx = []

        for itrain_index, itest_index in zip(train_list, test_list):
            train_index = df[df["animal_id"].isin(itrain_index)]["sample_idx"].values.astype(int)
            test_index = df[df["animal_id"].isin(itest_index)]["sample_idx"].values.astype(int)
            if self.verbose:
                print("TRAIN:", train_index, "TEST:", test_index, "TEST_ID:", df[df["animal_id"].isin(itest_index)]["animal_id"].values)
            training_idx.append(train_index)
            testing_idx.append(test_index)

        self.nfold = len(training_idx)
        print(
            "Bootstrap could build %d unique folds. stratification=%s"
            % (self.nfold, self.stratified)
        )

        for train_index, test_index in zip(training_idx, testing_idx):
            if self.verbose:
                print("TRAIN:", train_index, "TEST:", test_index, "TEST_ID:", df[df["sample_idx"].isin(test_index)]["animal_id"].values)
            yield train_index, test_index


if __name__ == "__main__":
    print("***************************")
    print("CUSTOM SPLIT TEST")
    print("***************************")