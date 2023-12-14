import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array


np.random.seed(0)


def inverse_anscombe(arr, sigma_sq=0, m=0, alpha=1, method='closed-form'):
    """
    Inverse of the Generalized Anscombe variance-stabilizing
    transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)


    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param m: mean of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :param method: 'closed_form' applies the closed-form approximation
    of the exact unbiased inverse. 'asym' applies the asymptotic
    approximation of the exact unbiased inverse.
    :return: inverse variance-stabilized array
    """
    sigma_sq /= alpha ** 2

    if method == 'closed-form':
        # closed-form approximation of the exact unbiased inverse:
        arr_trunc = np.maximum(arr, 0.8)
        inverse = ((arr_trunc / 2.) ** 2 + 0.25 * np.sqrt(1.5) * arr_trunc ** -1 - (11. / 8.) * arr_trunc ** -2 +
                   (5. / 8.) * np.sqrt(1.5) * arr_trunc ** -3 - (1. / 8.) - sigma_sq)
    elif method == 'asym':
        # asymptotic approximation of the exact unbiased inverse:
        inverse = (arr / 2.) ** 2 - 1. / 8 - sigma_sq
        # inverse = np.maximum(0, inverse)
    else:
        raise NotImplementedError('Only supports the closed-form')

    if alpha != 1:
        inverse *= alpha

    if m != 0:
        inverse += m

    return inverse


def anscombe(arr, sigma_sq=0, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)
    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :return: variance-stabilized array
    """
    v = np.maximum((arr / alpha) + (3. / 8.) + sigma_sq / (alpha ** 2), 0)
    f = 2. * np.sqrt(v)
    return f


class Sqrt(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse='csr')
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        X = check_array(X, accept_sparse='csr')
        X = np.sqrt(X)
        return X


class Log(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse='csr')
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        X = check_array(X, accept_sparse='csr')
        # X = np.log(X)
        X = np.ma.log(X)
        X.filled(0)
        return X

    def inverse_transform(self, X):
        return np.exp(X)


class Anscombe(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        self._validate_data(X, accept_sparse='csr')
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        X = check_array(X, accept_sparse='csr')
        return anscombe(X)

    def inverse_transform(self, X):
        return inverse_anscombe(X)


def create_synthetic(activity):
    pure = activity
    noise = np.random.normal(0, 1, len(activity))
    signal = pure + noise
    synt = signal * np.random.uniform(0.1, 3)
    synt[synt < 0] = 0
    return synt.astype(np.float16)


def plot_data(X, title="Activity sample before quotient normalisation"):
    # fig = make_subplots(rows=len(transponders), cols=1)
    fig = make_subplots(rows=1, cols=1)
    for i, sample in enumerate(X):
        timestamp = np.array(list(range(len(sample))))
        fig.append_trace(go.Scatter(
            name="sample %d" % i,
            opacity=.8,
            mode='lines',
            x=timestamp,
            y=sample,
        ), row=1, col=1)
    fig.update_layout(yaxis_range=[0, 20])
    fig.update_layout(title_text=title)
    #fig.show()


def create_synthetic_activity_data(n_samples=4):
    print("createSyntheticActivityData")
    samples_path = "C:/Users/fo18103/PycharmProjects/cats/src/dataset/norm_False_thresh_120/activity_cat_0_d_1_1min.csv"
    df = pd.read_csv(samples_path, header=None)
    df = df.fillna(0)
    crop = -4 - int(df.shape[1]/1.1)
    activity = df.iloc[10, : crop].values

    dataset = []
    for j in range(n_samples):
        A = create_synthetic(activity)
        dataset.append(A)

    return dataset


if __name__ == "__main__":
    print("********QuotientNormalizer*********")
    df = pd.DataFrame([[4, 1, 2, 2], [1, 3, 0, 3],
         [0, 7, 5, 1], [2, 0, 6, 8],
         [1, 6, 5, 4], [1, 2, 0, 4]])
    X = df.values
    print("X=", X)

    X_normalized = Anscombe().transform(X)
    # plotData(X, title="Activity sample before quotient normalisation")
    # plotData(X_normalized, title="Activity sample after quotient normalisation")

    print("after normalisation.")
    print(X_normalized)
    print("************************************")

    X = create_synthetic_activity_data()
    plot_data(X, title="Activity sample before quotient normalisation")

    X_normalized = Anscombe().transform(X)

    plot_data(X_normalized, title="Activity sample after quotient normalisation")


