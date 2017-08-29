# coding=utf-8
import numpy as np
import pandas as pd


def slice_array(a, window, stepsize=1):
    """
    Нарезает массив на последовательность массивов длины перемещаемого окна.

    :param a: исходный массив данных
    :param window: ширина окна
    :param stepsize: шаг сдвига окна
    :return: двумерный массив данных, в строках элемены полученные сдвигом окна
    """
    n = a.shape[0]
    return np.reshape(np.hstack( a[i : (i + window)] for i in range(0, n - window - 1, stepsize) ), (-1, window))


def load_prices(file, field, start, end):
    return np.array(pd.read_csv(file)[field])[start:end]


def load_window_returns(file, field, start, n, w_size, rtype="log", continues=True):
    all_prices = np.array(pd.read_csv(file)[field])[start - 1:start + n + w_size + 1]
    price_windows = slice_array(all_prices[1:], w_size)
    prev_prices = np.tile(all_prices[:all_prices.shape[0] - w_size - 2], (w_size, 1)).transpose() if continues \
        else slice_array(all_prices[:-1], w_size)
    if rtype == "log":
        return 100 * (np.log(price_windows) - np.log(prev_prices))
    elif rtype == "simple":
        return price_windows / prev_prices - 1
    elif rtype == "points":
        return price_windows - prev_prices


def load_returns(file, field, start, n, rtype="log"):
    all_prices = np.array(pd.read_csv(file)[field])[start - 1:start + n]
    cur_prices = all_prices[1:]
    prev_prices = all_prices[:all_prices.shape[0] - 1]
    if rtype == "log":
        return 100 * (np.log(cur_prices) - np.log(prev_prices))
    elif rtype == "simple":
        return cur_prices / prev_prices - 1
    elif rtype == "points":
        return cur_prices - prev_prices