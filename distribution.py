# coding=utf-8

import numpy as np

from scipy.optimize import minimize

from profiling import timeit

import os, ctypes
from scipy import integrate, LowLevelCallable

stdisplib = ctypes.CDLL(os.path.abspath('lib/stdisplib.so'))
stdisplib.f.restype = ctypes.c_double
stdisplib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)


class DistributionParams(ctypes.Structure):
    """
    Структура с параметрами распределения для совместимости с низкоуровневой реализацией.
    """
    _fields_ = [("n", ctypes.c_int),
                ("params", ctypes.POINTER(ctypes.c_double))]


class EvolvingProbabilityDistribution:
    """
    Базовый класс для модели нестационарной плотности вероятности.
    """

    def __init__(self, params, bounds):
        self.params = params
        self.interval = bounds
        self.status = -1

    def pdf(self, xs, params=None, t=np.inf):
        """
        Для заданных точек производит рассчет значений PDF.
        
        :param xs: точки, в которых необходимо посчитать значения
        :param params: параметры распределения
        :param t: 
        :return: 
        """
        raise NotImplementedError

    def cdf(self, xs, params=None, t=np.inf, left_bound=None):
        raise NotImplementedError

    @timeit
    def fit(self, X):
        """
        Оптимизирует параметры модели методом максимального правдоподобия.

        :param X: наблюдаемые данные, используемые для оценки модели
        :return: обученную модель плотности вероятности
        """

        result = minimize(
            self.__inv_log_likelihood, self.params, args=(X,),
            constraints=tuple(self.create_constraints(X)), options={'disp': True, 'maxiter': 1000},
            bounds=self.create_bounds(self.params), tol=5e-06
        )

        self.params = result.x
        self.status = result.status
        a, b = self.integration_interval(X)
        self.interval = (a, b)
        return self

    def create_constraints(self, X):
        """
        Создает список ограничений для данной модели. Метод может быть переопределен для конкретной модели.
        
        :param X: наблюдаемые данные, используемые для оценки модели
        :return: список ограничений
        """
        a, b = self.integration_interval(X)
        constraints = [
            # интеграл от функции PDF должен быть равен единице
            {
                'type': 'eq',
                'fun': lambda pars: self.cdf(np.array([b]), pars, left_bound=a) - 1
            },
            {
                # прижимаем концы стационарного распределения
                'type': 'ineq',
                'fun': lambda pars: 0.01 - self.pdf(np.array([a, b]), pars)
            }
        ]
        return constraints

    def create_bounds(self, pars):
        """
        Создает список ограничений на параметры распределения.
        
        :param pars: параметры распределения
        :return: список ограничений
        """
        return [(None, None) for i in range(0, params.shape[0])]

    def integration_interval(self, X):
        """
        Определяет границы интегрирования для рассчета полной вероятности.

        :return: границы интегрирования в виде пары значений
        """
        return np.max(X) - 0.5 * np.std(X), 0.5 * np.max(X) + np.std(X)

    def plot(self):
        raise NotImplementedError

    def __inv_log_likelihood(self, params, xs):
        """
        Производит рассчет значения функции правдоподобия (логарифм совместной вероятности) заданных 
        наблюдаемых значений при условии заданных параметров.
        
        :param params: параметры распределения
        :param xs: наблюдаемые значения
        :return: значение функции правдоподобия с обратным знаком (удобно для нужд оптимизации)
        """
        return -np.sum(np.log(self.pdf(xs, params, 0)))


class FpdFromHeatDiffusion(EvolvingProbabilityDistribution):
    """
    Модель распределения, которое подчиняется уравнению полученного как решение уравнения Фоккера-Планка методом 
    замены переменной через решение уравнения тепловой диффузии.
    """

    def __init__(self, params, bounds=(None, None)):
        EvolvingProbabilityDistribution.__init__(self, params, bounds)

    def pdf(self, xs, pars=None, t=np.inf):
        d_params = pars if pars is not None else self.params
        if np.isinf(t):
            return self.__stationary_pdf(xs, d_params)
        else:
            return self.__stationary_pdf(xs, d_params) * self.__pdf_perturbation(xs, d_params, t)

    def cdf(self, xs, pars=None, t=np.inf, left_bound=None):
        leftmost = left_bound if left_bound is not None else self.integration_interval(xs)[0]
        if np.isinf(t):
            return self.__optimized_stationary_cdf(xs, pars, leftmost)
        else:
            # TODO реализовать CDF для общего случая (можно без оптимизаций)
            raise NotImplementedError

    def __optimized_stationary_cdf(self, xs, pars, leftmost=None):
        """
        Оптимизированный метод для рассчета CDF для стационарного распределения, используется 
        низкоуровневая реализация функции для рассчета значения стационарного распределения (ускорение ~15x !!!).
        Для массива заданных точек CDF считается как сумма интегралов на отрезках между ними, так получается небольшое
        ускорение в рассчете 
        
        :param xs: точки, для которых необходимо посчитать значение CDF
        :param pars: параметры распределения
        :param leftmost: левая граница интегрирования
        :return: 
        """
        prev_cdf = 0
        result = np.zeros(xs.shape[0])
        indexes = np.argsort(xs)
        left_bound = leftmost if leftmost is not None else self.interval[0]

        dp = DistributionParams()
        dp.n = pars.shape[0] - 2
        dp.params = pars[2:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        user_data = ctypes.cast(ctypes.pointer(dp), ctypes.c_void_p)

        pdf_f = LowLevelCallable(stdisplib.f, user_data)

        for i in range(indexes.shape[0]):
            prev_cdf += integrate.quad(pdf_f, left_bound, xs[indexes[i]])[0]
            result[indexes[i]] = prev_cdf
            left_bound = xs[indexes[i]]
        return result

    def integration_interval(self, X):
        return 0, np.exp(np.max(np.log(X)) + 0.5 * np.std(np.log(X)))

    def create_bounds(self, pars):
        bounds = super().create_bounds(pars)
        bounds[0] = (0.2, 0.8)
        bounds[1] = (0.0, None)
        bounds[2] = (0.0, None)
        return bounds

    def __stationary_pdf(self, points, params):
        K = params[2]
        st_params = params[3:]
        powers = np.arange(st_params.shape[0]) + 1
        return K * np.exp(
            np.power(np.tile(points, (powers.shape[0], 1)), np.tile(powers, (points.shape[0], 1)).transpose())
                .transpose().dot(st_params)
        )

    def __pdf_perturbation(self, points, params, t):
        A = params[0]
        k = params[1]
        return 1 + A * np.exp(-np.power(k, 2) * t) * np.sin(k * (0.5 - self.__optimized_stationary_cdf(points, params, leftmost=0)))


import pandas as pd
from data_utils import slice_array

# читаем данные из файла
data = pd.read_csv("data/IB-EURUSD-8-XI-2016-2.csv")
prices = np.array(data["ask_close"])

k = 12
start = 42000
end = 42543

log_prices_shifted = np.log(prices[start:end - k - 2])
log_prices = np.log(prices[start + 1:end])

sliced_prices = slice_array(log_prices, k)

returns = 100 * (sliced_prices - np.tile(log_prices_shifted, (k, 1)).transpose())

params = np.zeros(7)
params[0] = 1.0 / 3
params[1] = 3 * np.pi
params[2] = 1

for i in range(0, returns.shape[0]):
    print("Iteration %d" % i)
    params = np.zeros(8)
    params[0] = 1.0 / 3
    params[1] = 3 * np.pi
    params[2] = 1
    returns_window = np.exp(returns[i, :])
    print(returns_window)
    fpd = FpdFromHeatDiffusion(params)
    fpd.fit(returns_window)
    params = fpd.params
    print(fpd.params, fpd.interval)
