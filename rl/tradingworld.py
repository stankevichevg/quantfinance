import numpy as np
from gym.spaces import Box

import logging.config

logging.config.fileConfig('logging.conf')

logger = logging.getLogger('trading_world')


class TradingWorld:
    """
    Представляет торговую сессию для обучения агента. Хранит текущее состояние, изменяемое как результат 
    совершения торгового действия.
    """
    def __init__(self, market_states, points, spread=5, max_allowed_position=3):
        """
        Инициализация торговой сессии.
        
        :param market_states: состояния рынка
        :param spread: спред для учета комиссии
        :param max_allowed_position: максимальная доступная позиция
        """
        self.market_states = market_states
        self.points = points
        self.position = 0
        self.cur_index = 0
        self.spread = spread
        self.max_allowed_position = max_allowed_position

    def sample(self):
        r_index = np.random.choice(self.market_states.shape[0])
        r_position = np.random.choice(self.max_allowed_position * 2 + 1) - self.max_allowed_position
        return self.state(index=r_index, position=r_position)

    def state(self, index=None, position=None):
        ind = index if index is not None else self.cur_index
        pos = position if position is not None else self.position
        market_state = self.market_states[ind]
        world_state = np.zeros((market_state.shape[1], market_state.shape[1]))
        world_state[2:,] = market_state

        block_size = int(market_state.shape[1] / (self.max_allowed_position * 2 + 1) - 1)

        raw_block_position = int(-(block_size / 2) + pos * block_size + int(market_state.shape[1] / 2))
        block_position = int(np.clip([raw_block_position], 0, market_state.shape[1] - block_size)[0])
        block = np.ones((2, block_size))

        world_state[0:2, block_position:(block_position + block_size)] = block

        return world_state

    def reset(self, position=0, cur_index=0):
        self.position = position
        self.cur_index = cur_index
        logger.debug("Reset environment: position(%d), cur_index(%d)" % (self.position, self.cur_index))
        logger.debug("Return state for index %d" % self.cur_index)
        return self.state(index=self.cur_index)

    def step(self, action):
        logger.debug("Doing the next step in the environment with action %f.", action)
        trade = action

        pos = abs(self.position + action)
        if pos > self.max_allowed_position:
            if action < 0:
                trade = action + (pos - self.max_allowed_position)
            else:
                trade = action - (pos - self.max_allowed_position)
            # self.cur_index += 1
            # return self.state(), -1000.0, True

        logger.debug("Execution trade of the %d lots." % trade)
        closed_positions = 0
        if self.position > 0 and trade < 0 or self.position < 0 and trade > 0:
            closed_positions = min(np.abs(self.position), np.abs(trade))
        commission = closed_positions * self.spread
        self.position += trade
        logger.debug("Transaction cost of %d points was paid." % commission)
        logger.debug("Current position is %d lots." % self.position)

        points = 100000 * self.points[self.cur_index]
        logger.debug("Target amount is %d points for current index %d." % (points, self.cur_index))
        reward = points * self.position - commission
        logger.debug("Current reward is %d points." % reward)

        self.cur_index += 1

        done = self.cur_index == (self.market_states.shape[0] - 1)
        return self.state(), reward, done
