# coding=utf-8
import multiprocessing
from multiprocessing import Pool

import numpy as np

from scipy.optimize import minimize

from profiling import timeit

import pandas as pd

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

    def __init__(self, params, interval, status):
        self.params = params
        self.interval = interval
        self.status = status

    def pdf(self, xs, params=None, t=np.inf, time_shifts=0.0):
        """
        Для заданных точек производит рассчет значений PDF.
        
        :param xs: точки, в которых необходимо посчитать значения
        :param params: параметры распределения
        :param t: момент времени
        :return: массив значений PDF в заданных точках в заданный момент времени
        """
        raise NotImplementedError

    def cdf(self, xs, params=None, t=np.inf, left_bound=None):
        """
        Для заданных точек производит рассчет значений CDF.
        
        :param xs: точки, в которых необходимо посчитать значения
        :param params: параметры распределения
        :param t: момент времени
        :param left_bound: левая граница интегрирования для нахождения CDF в каждой точке
        :return: массив значений CDF в заданных точках в заданный момент времени
        """
        raise NotImplementedError

    def moment(self, k, t=np.inf):
        """
        Производит рассчет k-ого момента данного распределения
        
        :param k: порядок момента
        :param t: момент времени
        :return: значение момента данного распределения в заданное время
        """
        return integrate.quad(lambda x: np.power(x, k) * self.pdf(np.array([x]), t=t), self.interval[0], self.interval[1])[0]

    def cme(self, a, b, t=np.inf):
        """
        Производит рассчет условного математического ожидания наблюдаемого значения при условии, 
        что оно лежит в интервале [a, b].
        
        :param b: пороговое значение
        :param t: момент времени
        :return: значение условного математического ожидания
        """
        return integrate.quad(lambda x: x * self.pdf(np.array([x]), t=t), a, b)[0]

    @timeit
    def fit(self, X):
        """
        Оптимизирует параметры модели методом максимального правдоподобия.

        :param X: наблюдаемые данные, используемые для оценки модели
        :return: обученную модель плотности вероятности
        """

        result = minimize(
            self.__inv_log_likelihood, self.params, args=(X,),
            constraints=tuple(self.create_constraints(X)), options={'disp': False, 'maxiter': 1000},
            bounds=self.create_bounds(self.params), tol=5e-06
        )

        self.params = result.x
        self.status = (result.status, result.fun)
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
        return [(None, None) for i in range(0, pars.shape[0])]

    def integration_interval(self, X):
        """
        Определяет границы интегрирования для рассчета полной вероятности.

        :return: границы интегрирования в виде пары значений
        """
        return np.min(X) - 0.5 * np.std(X), np.max(X) + 0.5 * np.std(X)

    def plot(self):
        raise NotImplementedError

    def to_df(self):
        """
        Сохраняет данную модель распределения в DataFrame.
        
        :return: представление модели распределения в виде DataFrame
        """
        raise NotImplementedError

    @staticmethod
    def from_df(df):
        """
        Восстанавливает модель из представления в виде DataFrame.
        
        :param df: представление модели распределения в виде DataFrame
        :return: модель распределения или список моделей, если фрэйм содержит больше чем одну модель
        """
        raise NotImplementedError

    def __inv_log_likelihood(self, pars, xs):
        """
        Производит рассчет значения функции правдоподобия (логарифм совместной вероятности) заданных 
        наблюдаемых значений при условии заданных параметров.
        
        :param params: параметры распределения
        :param xs: наблюдаемые значения
        :return: значение функции правдоподобия с обратным знаком (удобно для нужд оптимизации)
        """
        time_shifts = (-np.arange(0, xs.shape[0], 1) / (365 * 24))[::-1]
        # time_shifts = 0.0
        return -np.sum(np.log(self.pdf(xs, pars, 0, time_shifts=time_shifts)))


class FpdFromHeatDiffusion(EvolvingProbabilityDistribution):
    """
    Модель распределения, которое подчиняется уравнению полученного как решение уравнения Фоккера-Планка методом 
    замены переменной через решение уравнения тепловой диффузии.
    """

    def __init__(self, pars, interval=(None, None), status=(-1, None)):
        EvolvingProbabilityDistribution.__init__(self, pars, interval, status)

    def pdf(self, xs, pars=None, t=np.inf, time_shifts=0.0):
        d_params = pars if pars is not None else self.params
        if np.isinf(t):
            return self.__stationary_pdf(xs, d_params)
        else:
            return self.__stationary_pdf(xs, d_params) * \
                   self.__pdf_perturbation(xs, d_params, t, time_shifts=time_shifts)

    def cdf(self, xs, pars=None, t=np.inf, left_bound=None):
        leftmost = left_bound if left_bound is not None else self.integration_interval(xs)[0]
        if np.isinf(t):
            return self.__optimized_stationary_cdf(xs, pars if pars is not None else self.params, leftmost)
        else:
            # TODO реализовать CDF для общего случая (можно без оптимизаций)
            return integrate.quad(lambda x: self.pdf(np.array([x]), t=t), self.interval[0], xs[0])[0]

    def to_df(self):
        values, names = [], []
        names.append('status')
        values.append(self.status[0])
        names.append('logl')
        values.append(self.status[1])
        names.append('left_bound')
        values.append(self.interval[0])
        names.append('right_bound')
        values.append(self.interval[1])
        names.append('A')
        values.append(self.params[0])
        names.append('k')
        values.append(self.params[1])
        names.append('K')
        values.append(self.params[2])
        for i, par in enumerate(self.params[3:].tolist()):
            names.append('lambda' + repr(i))
            values.append(par)
        return pd.DataFrame([values], columns=names)

    @staticmethod
    def from_df(df):
        models = []
        for _, row in df.iterrows():
            all_params = row.values
            model_params = all_params[4:]
            interval = (all_params[2], all_params[3])
            status = (all_params[0], all_params[1])
            models.append(FpdFromHeatDiffusion(model_params, interval, status))
        return models[0] if len(models) == 1 else models

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
        bounds[0] = (0.15, 0.85)
        bounds[1] = (0.0, None)
        bounds[2] = (0.0, None)
        return bounds

    def __stationary_pdf(self, points, pars):
        K = pars[2]
        st_params = pars[3:]
        powers = np.arange(st_params.shape[0]) + 1
        return K * np.exp(
            np.power(np.tile(points, (powers.shape[0], 1)), np.tile(powers, (points.shape[0], 1)).transpose())
                .transpose().dot(st_params)
        )

    def __pdf_perturbation(self, points, pars, t, time_shifts=0.0):
        A = pars[0]
        k = pars[1]
        return 1 + A * np.exp(-np.power(k, 2) * (t + time_shifts)) * np.sin(k * (0.5 - self.__optimized_stationary_cdf(points, pars, leftmost=0)))

