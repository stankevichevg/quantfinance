import pandas as pd
import numpy as np

from datautils import load_returns
from distribution import FpdFromHeatDiffusion
from preprocessing import models_data_file, create_features, prices_data_file, models_features_data_file

if __name__ == '__main__':

    # df1 = pd.DataFrame.from_csv("data/.temp/models_EURUSD_s41770_n200_w12_log.csv", sep=";")
    # df2 = pd.DataFrame.from_csv("data/.temp/models_EURUSD_s41970_n600_w12_log.csv", sep=";")
    #
    # df = df1.append(df2, ignore_index=True)
    #
    # complexity = 7
    # window = 12
    # start = 41770
    # N = 800
    # instrument = "EURUSD"
    # field = "bid_close"
    #
    # df.to_csv(models_data_file(instrument, start, N, window))
    #
    # models = FpdFromHeatDiffusion.from_df(df)
    #
    # points = load_returns(prices_data_file(instrument), field, start + window, N, rtype="points")
    #
    # features_df = create_features(models, points, 4)
    #
    # features_df.to_csv(models_features_data_file(instrument, start, N, window))

    # market_states = pd.DataFrame.from_csv("data/.temp/models_features_EURUSD_s41450_n300_w12_log.csv")[:5000]
    market_states = pd.DataFrame.from_csv("data/.temp/models_features_EURUSD_s38470_n100_w12_log.csv")[:5000]

    # market_states['signal'] = market_states['mean_st'] - market_states['mean']
    market_states['signal'] = market_states['mean_st'] - market_states['mean_st'].shift()

    market_states['signal'] = market_states['signal'].apply(lambda x: 0 if np.abs(x) > 1 else x)
    market_states['signal_sgn'] = market_states['signal'].apply(lambda x: -1.0 if x < 0.00 else 1.0 if x > 0.0 else 0.0)

    market_states['target_traded'] = market_states['target'] * market_states['signal_sgn']

    market_states['signal'] = market_states['mean_st'] - market_states['mean_st'].shift()
    # market_states['signal'] = market_states['mean_1h'] - 1
    market_states['mean_st'] = market_states['mean_st'].apply(lambda x: 0.0 if np.abs(x) > 10 else x)
    market_states['signal'] = market_states['signal'].apply(lambda x: 0.0 if np.abs(x) > 1 else x)
    market_states['signal_sgn'] = market_states['signal'].apply(lambda x: -1.0 if x < 0.0 else 1.0)
    market_states['target_traded'] = market_states['target'] * market_states['signal_sgn']
    market_states['target_traded'].cumsum().plot()
    # (market_states['target']).cumsum().plot()
    (market_states['signal_sgn'] / 1000).plot()

    (market_states['target']).cumsum().plot()

    print("")
