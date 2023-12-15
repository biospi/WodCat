from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from utils._anscombe import anscombe

np.random.seed(0)


def normalize(X, out_dir, output_graph, enable_qn_peak_filter, animal_ids, labels):
    out_dir_ = out_dir / "_normalisation"
    traces = []
    X = X.astype(np.float)
    X_o = X.copy()

    if enable_qn_peak_filter:
        X_peak_mask = X.copy()
        X_peak_mask[:] = np.nan
        n_peak = int(np.ceil(X.shape[1]/(4*60)))
        stride = int(X.shape[1] / n_peak / 2)
        w = 1

        if n_peak == 1:
            for i in range(X.shape[0]):
                    X_peak_mask[i, stride - w:stride + w+1] = X[i, stride - w:stride + w+1]
        else:
            for i in range(X.shape[0]):
                cpt = 0
                for j in range(stride, X.shape[1], stride):
                    cpt += 1
                    if cpt % 2 == 0:
                        continue
                    #print(j)
                    X_peak_mask[i, j-w:j+w] = X[i, j-w:j+w]
        X = X_peak_mask

    zmin, zmax = np.nanmin(np.log(anscombe(X))), np.nanmax(np.log(anscombe(X)))
    if np.isinf(zmin) or np.isnan(zmin):
        zmin = 0
    if np.isinf(zmax) or np.isnan(zmax):
        zmax = 1
    # zmin, zmax = None, None

    if output_graph:
        traces.append(
            plotHeatmap(
                zmin,
                zmax,
                np.array(X).copy(),
                out_dir_,
                "STEP 0 | Samples",
                "0_X_samples.html",
                y_log=True,
                xaxis_title="Time (in minutes)",
                yaxis_title="Samples"
            )
        )

    # step 1 find pointwise median sample [median of col1, .... median of col n].
    median_array = np.median(X, axis=0)
    median_array[median_array <= 0] = 1
    if output_graph:
        traces.append(
            plotLine(
                [median_array],
                out_dir_,
                "STEP 1 | find pointwise median sample [median of col1, .... median of col n]",
                "1_median_array.html",
                xaxis_title = "Time (in minutes)",
                yaxis_title = "Median (of samples features)"
            )
        )
    # cwt_power(median_array, out_dir, step_slug="HERD", avg=np.average(median_array))

    # step 2 divide each sample by median array keep div by 0 as NaN!!
    X_median = []
    for x in X:
        div = np.divide(x, median_array)
        div[div == -np.inf] = np.nan
        div[div == np.inf] = np.nan
        div[div <= 0] = np.nan
        X_median.append(div)
    if output_graph:
        traces.append(
            plotHeatmap(
                zmin,
                zmax,
                np.array(X_median).copy(),
                out_dir_,
                "STEP 2 | divide each sample by median array\n "
                "keep div by 0 as NaN, set 0 to NaN",
                "2_X_median.html",
                y_log=True,
                xaxis_title="Time (in minutes)",
                yaxis_title="Samples"
            )
        )

    # step 3 Within each sample (from iii) store the median value of the sample(excluding 0 value!), which will produce an array of
    # median values (1 per samples).
    within_median = []
    for msample in X_median:
        clean_sample = msample[~np.isnan(msample)]
        m = np.median(clean_sample)
        if m <= 0:
            m = 1
        within_median.append(m)

    if output_graph:
        traces.append(
            plotLine(
                [within_median],
                out_dir_,
                "STEP 3 | Within each sample (rows from step2) store the median"
                " value of the sample,\n which will produce an array of median "
                "values (1 per samples)",
                "3_within_median.html",
                x_axis_count=True,
                y_log=False,
                xaxis_title="Time (in minutes)",
                yaxis_title="Samples"
            )
        )

    # step 4 Use the array of medians to scale(divide) each original sample, which will give all quotient normalized samples.
    qnorm_samples = []
    for i, (s, s_o) in enumerate(zip(X, X_o)):
        s[np.isnan(s)] = s_o[np.isnan(s)]
        q_sample = np.divide(s, within_median[i])
        qnorm_samples.append(q_sample)

    if output_graph:
        # qnorm_samples = []
        # for l in labels:
        #     qnorm_samples.append([l])
        animal_mask = (animal_ids[:-1] != animal_ids[1:])
        animal_mask = np.append(animal_mask, False)
        qnorm_samples_mask = []
        for i, sample in enumerate(qnorm_samples):
            if animal_mask[i]:
                for _ in range(5):
                    qnorm_samples_mask.append([np.nan] * (len(sample)+1))
            # sample[::] = np.nan
            sample = np.append(sample, labels[i])
            qnorm_samples_mask.append(sample)
        qnorm_samples_mask = np.vstack(qnorm_samples_mask)
        traces.append(
            plotHeatmap(
                zmin,
                zmax,
                np.array(qnorm_samples_mask).copy(),
                out_dir_,
                "STEP 4 | Use the array of medians"
                " to scale(divide) each original sample,\n"
                " which will give all quotient normalized samples.",
                "4_qnorm_sample.html",
                y_log=True,
                xaxis_title="Time (in minutes)",
                yaxis_title="Samples",
            )
        )

    # step 5 substract step 4 from step 1
    diff = np.log(anscombe(np.array(qnorm_samples))) - np.log(anscombe(median_array))

    if output_graph:
        traces.append(
            plotHeatmap(
                zmin,
                zmax,
                diff,
                out_dir_,
                "STEP 5 | Substract step 4 (quotient normalised samples)\n"
                " from step 1 (median array)",
                "5_diff.html",
                y_log=False,
                xaxis_title="Time (in minutes)",
                yaxis_title="Samples"
            )
        )

    if output_graph:
        plot_all(traces, out_dir_, title="Quotient Normalisation 5 STEPS")

    df_norm = np.array(qnorm_samples)
    return df_norm


class CenterScaler(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        norm="q",
        *,
        out_dir=None,
        copy=True,
        center_by_sample=False,
        divide_by_std=False
    ):
        self.out_dir = out_dir
        self.norm = norm
        self.copy = copy
        self.center_by_sample = center_by_sample
        self.divide_by_std = divide_by_std

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        """Center data

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        # copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr')
        if self.center_by_sample:
            if not isinstance(X, np.ndarray):
                X = np.array(X)

            X_centered = np.ones(X.shape)
            X_centered[:] = np.nan
            for i in range(X.shape[0]):
                a = X[i, :]
                X_centered[i, :] = a - np.average(a)
                if self.divide_by_std:
                    X_centered[i, :] = X_centered[i, :] / np.std(X_centered[i, :])
        else:
            X_centered = X - np.average(X)
            if self.divide_by_std:
                X_centered = X_centered / np.std(X_centered)

        return X_centered


class QuotientNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, norm="q", *, out_dir=None, copy=True, output_graph=False, enable_qn_peak_filter=False, animal_ids=None, labels=None):
        self.out_dir = out_dir
        self.norm = norm
        self.copy = copy
        self.output_graph = output_graph
        self.enable_qn_peak_filter = enable_qn_peak_filter
        self.animal_ids = animal_ids
        self.labels = labels

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        """QN

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        # copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse="csr")
        norm = normalize(X, self.out_dir, self.output_graph, self.enable_qn_peak_filter, self.animal_ids, self.labels)
        # norm_simple = normalize_simple(X, self.out_dir)
        return norm


def create_synthetic(activity):
    pure = activity
    noise = np.random.normal(0, 200, len(activity))
    signal = pure + noise
    synt = signal * np.random.uniform(0.1, 1.5)
    synt[synt < 0] = 0
    return synt.astype(int)


def plot_all(
    traces,
    out_dir,
    title="Quotient Normalisation STEPS",
    filename="steps.html",
    simple=False,
):
    ts = []
    for trace in traces:
        ts.append(trace[1])
    fig = make_subplots(rows=len(traces), cols=1, subplot_titles=tuple(ts))
    for i, trace in enumerate(traces):
        fig.append_trace(trace[0], row=i + 1, col=1)

    fig.update_layout(title_text=title)
    fig.update_layout(showlegend=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))


def plotLine(
    X, out_dir="", title="title", filename="file.html", x_axis_count=False, y_log=False,
    xaxis_title=None, yaxis_title=None
):
    # fig = make_subplots(rows=len(transponders), cols=1)
    fig = make_subplots(rows=1, cols=1)
    for i, sample in enumerate(X):
        timestamp = get_time_ticks(len(sample))
        if x_axis_count:
            timestamp = list(range(len(timestamp)))
        if y_log:
            sample_log = np.log(sample)
        trace = go.Scatter(
            opacity=0.8,
            mode='lines',
            x=timestamp,
            y=sample_log if y_log else sample,
        )
        fig.append_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    return trace, title


def get_time_ticks(nticks):
    date_string = "2012-12-12 00:00:00"
    Today = datetime.fromisoformat(date_string)
    date_list = [Today + timedelta(minutes=1 * x) for x in range(0, nticks)]
    # datetext = [x.strftime('%H:%M') for x in date_list]
    return date_list


def plotHeatmap(
    zmin, zmax, X, out_dir="", title="Heatmap", filename="heatmap.html", y_log=False,
xaxis_title=None, yaxis_title=None
):
    # fig = make_subplots(rows=len(transponders), cols=1)
    # ticks = get_time_ticks(X.shape[1])
    X = X + 0
    ticks = list(range(X.shape[1]))
    fig = make_subplots(rows=1, cols=1)
    if y_log:
        X_log = np.log(anscombe(X))
    if zmin is None:
        trace = go.Heatmap(
            z=X_log if y_log else X,
            x=ticks,
            y=list(range(X.shape[0])),
            colorscale="Viridis",
        )
    else:
        trace = go.Heatmap(
            z=X_log if y_log else X,
            x=ticks,
            y=list(range(X.shape[0])),
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
        )
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    return trace, title


def createSyntheticActivityData(n_samples=4):
    print("createSyntheticActivityData")
    samples_path = (
        "F:/Data2/dataset_gain_7day/activity_delmas_70101200027_dbft_7_1min.csv"
    )
    df = pd.read_csv(samples_path, header=None)
    df = df.fillna(0)
    crop = -4 - int(df.shape[1] / 1.1)
    activity = df.iloc[259, 9353 : 9353 + 60 * 6].values

    dataset = []
    for j in range(n_samples):
        A = create_synthetic(activity)
        dataset.append(A)

    return dataset


if __name__ == "__main__":
    print("********QuotientNormalizer*********")
    df = pd.DataFrame(
        [
            [4, 1, 2, 2],
            [1, 3, 0, 3],
            [0, 7, 5, 1],
            [2, 0, 6, 8],
            [1, 6, 5, 4],
            [1, 2, 0, 4],
        ]
    )
    X = df.values
    print("X=", X)
    X_centered = CenterScaler().transform(X)

    out_dir = "F:/Data2/_normalisation_1"
    X_normalized = QuotientNormalizer(out_dir=out_dir).transform(X)
    # plotData(X, title="Activity sample before quotient normalisation")
    # plotData(X_normalized, title="Activity sample after quotient normalisation")

    print("after normalisation.")
    print(X_normalized)
    print("************************************")

    out_dir = "F:/Data2/_normalisation_2"
    X = createSyntheticActivityData()
    plotLine(X, out_dir=out_dir, title="Activity sample before quotient normalisation")

    X_normalized = QuotientNormalizer(out_dir=out_dir).transform(X)

    plotLine(
        X_normalized,
        out_dir=out_dir,
        title="Activity sample after quotient normalisation",
    )
    print()
