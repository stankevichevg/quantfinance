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