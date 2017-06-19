import pandas as pd
import numpy as np

from datautils import load_returns
from distribution import FpdFromHeatDiffusion
from preprocessing import models_data_file, create_features, prices_data_file, models_features_data_file

if __name__ == '__main__':

    market_states = pd.DataFrame.from_csv("data/.temp/models_features_EURUSD_s41970_n300_w12_log.csv")[:5000]

    market_states['signal'] = market_states['mean_st'] - market_states['mean_st'].shift()

    market_states['signal'] = market_states['signal'].apply(lambda x: 0 if np.abs(x) > 1 else x)
    market_states['signal_sgn'] = market_states['signal'].apply(lambda x: -1.0 if x < 0.00 else 1.0 if x > 0.0 else 0.0)

    market_states['target_traded'] = market_states['target'] * market_states['signal_sgn']

    market_states['signal'] = market_states['mean_st'] - market_states['mean_st'].shift()
    market_states['mean_st'] = market_states['mean_st'].apply(lambda x: 0.0 if np.abs(x) > 10 else x)
    market_states['signal'] = market_states['signal'].apply(lambda x: 0.0 if np.abs(x) > 1 else x)
    market_states['signal_sgn'] = market_states['signal'].apply(lambda x: -1.0 if x < 0.0 else 1.0)
    market_states['target_traded'] = market_states['target'] * market_states['signal_sgn']
    market_states['target_traded'].cumsum().plot()
    (market_states['signal_sgn'] / 1000).plot()

    (market_states['target']).cumsum().plot()
