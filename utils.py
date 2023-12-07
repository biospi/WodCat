import random


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
    v = np.maximum((arr / alpha) + (3.0 / 8.0) + sigma_sq / (alpha**2), 0)
    f = 2.0 * np.sqrt(v)
    return f


def attribute_color(df):
    is_week_day = df[6]
    is_daytime = df[7]
    activity = df[3]
    # if activity > 300:
    #     return '#ff0000'
    if is_week_day and is_daytime:
        return "#fed8b1"
    if is_week_day and not is_daytime:
        return "#ff8c00"
    if not is_week_day and is_daytime:
        return "#90ee90"
    if not is_week_day and not is_daytime:
        return "#013220"


def check_if_hour_daylight(hour):
    if (hour >= 8) and (hour < 20):
        return True
    else:
        return False


def random_permutations(input_list, k):
    result = []
    # Fill the reservoir with the first k elements
    for i in range(len(input_list)):
        if i < k:
            result.append(input_list[i])
        else:
            # Randomly choose an index to potentially replace
            j = random.randint(0, i)
            if j < k:
                result[j] = input_list[i]
    return result


def efficient_permutation(input_list, n_max, n_peak):
    idxs = [random_permutations(input_list, n_peak) for _ in range(n_max)]
    return idxs


if __name__ == "__main__":
    # Example usage with input list [1, 2, 3, ..., 1000000] and k = 10
    input_list = list(range(1, 1000001))
    k = 10
    selected_permutations = [random_permutations(input_list, 3) for _ in range(k)]

    print(selected_permutations)