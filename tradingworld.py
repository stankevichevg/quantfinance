import numpy as np
from gym.spaces import Box

features = np.array([
    'mean',
    'dispersion',
    # 'dispersion_st',
    # 'kurtosis',
    # 'kurtosis_st',
    # 'mean_st',
    # 'neg_cme',
    'pos_cdf',
    # 'pos_cdf_st',
    'pos_cme',
    # 'skewness',
    # 'skewness_st'
])

class TradingGameWorld:
    def __init__(self, market_states, spread=5, max_allowed_position=10, game_length=1000):
        self.market_states = market_states
        self.position = 0
        self.cur_index = 0
        self.game_length = game_length
        self.shift = 0
        self.action_space = Box(low=np.array([-1]), high=np.array([1]))
        self.spread = spread
        self.max_allowed_position = max_allowed_position

    def sample(self):
        r_index = np.random.choice(self.market_states.shape[0])
        r_position = np.random.choice(self.max_allowed_position * 2 + 1) - self.max_allowed_position
        return self.state(index=r_index, position=r_position)

    def state(self, index=None, position=None):
        ind = index if index is not None else self.cur_index
        pos = position if position is not None else self.position
        market_state = self.market_states.iloc[ind][features].values
        market_state2 = self.market_states.iloc[ind-1][features].values
        internal_state = np.array([pos])
        fs = np.concatenate((
            # market_state,
            market_state - market_state2,
            internal_state
        ), axis=0)
        return fs

    def reset(self, position=0, cur_index=0, shift=10):
        self.position = position
        self.cur_index = cur_index
        self.shift = shift
        return self.state(index=(self.shift + self.cur_index))

    def __define_lots(self, action):
        des = action[0]
        lots = 0
        boundary = 0.0
        if des > boundary:
            lots = 1
        elif des < -boundary:
            lots = -1
        return lots

    def step(self, action, scale_reward=True):
        lots_should_be = self.__define_lots(action)
        trade = lots_should_be - self.position
        closed_positions = 0
        if self.position > 0 and trade < 0 or self.position < 0 and trade > 0:
            closed_positions = min(np.abs(self.position), np.abs(trade))
        commission = closed_positions * self.spread

        self.cur_index += 1
        self.position += trade

        points = self.market_states.iloc[self.shift + self.cur_index]["target"]
        reward = (100000 * points) * self.position - commission

        # придаем последним вознаграждениям больший вес, чтобы агент
        # быстрее приспасабливался к изменяемым правилам игры
        if scale_reward:
            reward = reward * np.exp(-0.5 * (self.game_length - self.cur_index) / self.game_length)

        done = self.cur_index == (self.game_length - 1)

        return self.state(index=(self.shift + self.cur_index)), reward, done
