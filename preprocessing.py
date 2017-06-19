import multiprocessing
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from itertools import product

from datautils import load_returns, load_window_returns
from distribution import FpdFromHeatDiffusion

import logging.config

logging.config.fileConfig('logging.conf')

logger = logging.getLogger('default')

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # где не смогли определеить, выставляем по умолчанию


def fit_hd_model(observations, complexity=8):
    # 1. создаем сетку параметров
    # 0.1 <= k <= 0.95
    k_values = np.arange(0.3, 0.4, 0.1)
    # 2 * pi <= A <= 8 * pi
    a_values = np.pi * np.arange(3, 4, 1)
    # 2. каждую модель оптимизируем в каждом узле сетки
    pairs = list(product(k_values, a_values))
    # 3. выбираем лучшее решение из всех (максимальное значение правдоподобия)
    best_model = None
    best_pair = None
    best_score = np.inf
    mparams = np.zeros(complexity)
    for pair in pairs:
        mparams[0] = pair[0]
        mparams[1] = pair[1]
        mparams[2] = 1
        fpd = FpdFromHeatDiffusion(mparams)
        fpd_model = fpd.fit(observations)
        if fpd_model.status[1] < best_score:
            best_model = fpd_model
            best_pair = pair
            best_score = fpd_model.status[1]
    logger.debug("The best model have been found for init params: (%f, %f)", best_pair[0], best_pair[1])
    logger.debug("Model fitting status: %s", best_model.status)
    logger.debug("Found params: %s", best_model.params)

    return best_model

def create_models(train, complexity, n_threads=1):
    ms = []
    with Pool(processes=n_threads) as pool:
        results = [pool.apply_async(fit_hd_model, (train[i, :], complexity)) for i in range(train.shape[0])]
        for i, res in enumerate(results):
            ms.append(res.get())
            print("Model %d is ready" % i)
    return ms

def save_models(ms, file=None):
    df = None
    for m in ms:
        if df is None:
            df = m.to_df()
        else:
            df = df.append(m.to_df(), ignore_index=True)
    if file is not None:
        df.to_csv(file, sep=';')
    return df


def extract_model_features(m, p):
    return {
        # первый момент для стационарного распределения
        'mean_st': m.moment(1),
        # первый момент для текущего распределения
        'mean': m.moment(1, 0),
        # первый момент для распределения через час
        'mean_1h': m.moment(1, 1.0/(365 * 24)),
        # первый момент для распределения через два часа
        'mean_2h': m.moment(1, 2.0/(365 * 24)),
        # первый момент для распределения через два часа
        'mean_3h': m.moment(1, 3.0/(365 * 24)),
        # первый момент для распределения через два часа
        'mean_4h': m.moment(1, 4.0/(365 * 24)),
        # второй момент для стационарного распределения
        'dispersion_st': m.moment(2),
        # второй момент для текущего распределения
        'dispersion': m.moment(2, 0),
        # третий момент для текущего  распределения
        'skewness': m.moment(3, 0),
        # третий момент для стационарного распределения
        'skewness_st': m.moment(3),
        # четвертый момент для стационарного распределения
        'kurtosis_st': m.moment(4),
        # четвертый момент для текущего распределения
        'kurtosis': m.moment(4, 0),
        # вероятность, что наблюдаемое значение больше 1.0 для текущего распределения
        'pos_cdf': 1 - m.cdf(np.array([1.0]), t=0),
        # вероятность, что наблюдаемое значение больше 0.0 для стационарного распределения
        'pos_cdf_st': m.cdf(np.array([m.interval[1]]), left_bound=1.0)[0],
        # условное математическое ожидание наблюдения, при наблюдаемом значении > 1.0 для текущего распределения
        'pos_cme': m.cme(1.0, m.interval[1], t=0),
        # условное математическое ожидание наблюдения, при наблюдаемом значении < 1.0 для текущего распределения
        'neg_cme': m.cme(m.interval[0], 1.0),
        # количество пунктов, реализованное в следующем периоде
        'target': p
    }


def create_features(models, points, n_threads=4):
    df = pd.DataFrame()
    with Pool(processes=n_threads) as pool:
        results = [pool.apply_async(extract_model_features, (m, p)) for (m, p) in zip(models, points)]
        for i, res in enumerate(results):
            df = df.append(res.get(), ignore_index=True)
            print("Model %d is ready" % i)
    return df


def prices_data_file(instrument):
    return "data/IB-%s-8-XI-2016-2.csv" % (instrument)


def models_data_file(instrument, start, N, window):
    return "data/.temp/models_%s_s%s_n%s_w%s_log.csv" % (instrument, start, N, window)


def models_features_data_file(instrument, start, N, window):
    return "data/.temp/models_features_%s_s%s_n%s_w%s_log.csv" % (instrument, start, N, window)

if __name__ == '__main__':
    complexity = 7
    window = 12
    start = 41970
    N = 300
    instrument = "EURUSD"
    field = "bid_close"

    # print("Load data for %s" % (instrument))
    file_name = prices_data_file(instrument)
    w_returns = load_window_returns(file_name, field, start, N, window, continues=True)
    models = create_models(np.exp(w_returns), complexity, 2)
    df = save_models(models, models_data_file(instrument, start, N, window))

    # df = pd.DataFrame.from_csv("data/.temp/models_EUR_s28045_n5000_w12_log.csv", sep=";")
    models = FpdFromHeatDiffusion.from_df(df)[:N]

    points = load_returns(file_name, field, start + window, N, rtype="points")

    features_df = create_features(models, points, cpus)

    features_df.to_csv(models_features_data_file(instrument, start, N, window))