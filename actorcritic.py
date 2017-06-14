import itertools
import numpy as np
import tensorflow as tf
import collections

import pandas as pd
import sklearn.preprocessing

from tradingworld import TradingGameWorld

import logging.config

logging.config.fileConfig('logging.conf')

logger = logging.getLogger('default')

EpisodeStats = collections.namedtuple("Stats", ["episode_lengths", "episode_rewards"])

TradingResults = collections.namedtuple("TradingResults", ["strategy_points", "bnh_points", "mu", "sigma"])

class PolicyEstimator():
    """
    Апроксиматор функции политики.
    """

    def __init__(self, scaler, input_size, learning_rate=0.01, scope="policy_estimator"):
        self.scaler = scaler
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [input_size], "state")
            self.action = tf.placeholder(tf.float32, name="action")
            self.target = tf.placeholder(tf.float32, name="target")
            # Простая линейная классификация
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())
            self.mu = tf.squeeze(self.mu)
            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist.sample(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
            # Loss and train op
            self.loss = -self.normal_dist.log_pdf(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None, scale=True):
        sess = sess or tf.get_default_session()
        state = self.scaler.transform([state])[0]
        return sess.run([self.action, self.mu, self.sigma], {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = self.scaler.transform([state])[0]
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Апроксиматор функции ценности состояния.
    """

    def __init__(self, scaler, input_size, learning_rate=0.1, scope="value_estimator"):
        self.scaler = scaler

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [input_size], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # Пока простая линейная регрессия
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = self.scaler.transform([state])[0]
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = self.scaler.transform([state])[0]
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def play_episodes(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0, shift=0):
    """
        Актор-Критик алгоритм для обучения с подкреплением. Оптимизирует функцию стратегии градиентным методом.

        Args:
            env: игровая среда
            estimator_policy: апроксимация политики для оптимизации
            estimator_value: апроксимация ценности состояния используется как базовая стратегия
            num_episodes: число эпизодов для прогонки
            discount_factor: дисконтирующий фактор

        Returns:
            Статистика по эпизодам
        """

    # статистики, заполняется по мере прохождения эпизодов
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    logger.debug("Starting of playing episodes: N(%d)" % num_episodes)
    for i_episode in range(num_episodes):
        # сбрасываем состояние среды, получаем начальное состояние
        state = env.reset(shift=shift)
        episode = []
        rewards = np.zeros(3000)
        errors = np.zeros(3000)
        # Одна прогонка обучения
        for t in itertools.count():
            # Делаем очередной шаг стратегии
            decision = estimator_policy.predict(state)
            action = decision[0]
            logger.debug("Value prediction is %f for %dth step" % (action, t))
            next_state, reward, done = env.step(action, scale_reward=False)
            # Сохраняем переход
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))
            # Обновляем статистику эпизода
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            rewards[t] = reward
            # Рассчитываем целевое TD значение
            # value_next = estimator_value.predict(next_state) * np.exp(-0.5 * (env.game_length - env.cur_index) / env.game_length)
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            errors[t] = td_error
            # Обновляем апроксиматор ценности состояния
            estimator_value.update(state, td_target)
            # Обновляем апроксиматор политики
            estimator_policy.update(state, td_error, action)
            if done:
                break
            state = next_state

        logger.info("Episode {}/{} ({})".format(i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
        # logger.info("Squared error id %f", np.sum(np.power(errors, 2)))

    return stats


def actor_critic(env, estimator_policy, estimator_value,  discount_factor=1.0, step=15, length=5000):
    # заведем структурку, куда будем сохранять основные данные с прогрессом работы для последующего анализа
    results = TradingResults(
        strategy_points=np.zeros(length),
        bnh_points=np.zeros(length),
        mu=np.zeros(length),
        sigma=np.zeros(length)
    )
    start_shift = 1000
    # вначале нет открытых позиций
    position = 0
    # сбрасываем состояние игры на начальное
    env.reset(position=position)
    # Шаг 1) делаем первый прогон в 50 эпизодов (прогрев)
    play_episodes(env, estimator_policy, estimator_value, 25, discount_factor, shift=start_shift)
    for shift in range(step, length, step):
        # Шаг 2) сдвигаем на shift позиций
        logger.debug("Applying shift of %d positions" % shift)
        state = env.reset(position=position, shift=shift+start_shift, cur_index=env.game_length - step)
        # Шаг 3) предсказываем step следующих значений по смещению среднего, сохраняем результат этих step предсказаний
        for s in range(0, step):
            ind = [env.shift + env.cur_index]
            results.bnh_points[ind] = 100000 * env.market_states.iloc[env.shift + env.cur_index - 1]["target"]
            logger.debug("Predict action for the state %d" % ind[0])
            decision = estimator_policy.predict(state)
            mu = decision[1]
            sigma = decision[2]
            next_state, reward, done = env.step([decision[0]],scale_reward=False)
            state = next_state
            position = env.position
            results.strategy_points[ind] = reward
            results.mu[ind] = mu
            results.sigma[ind] = sigma
            logger.info((np.sum(results.strategy_points), decision[0], mu, sigma))
        # Шаг 4) дообучаем, делаем 10 прогонов
        play_episodes(env, estimator_policy, estimator_value, 5, discount_factor, shift=shift+start_shift)
        # Шаг 5) повторяем с шага 3

    return results


if __name__ == '__main__':

    market_states = pd.DataFrame.from_csv("data/.temp/models_features_EURUSD_s28045_n5000_w12_log.csv")
    env = TradingGameWorld(market_states, game_length=1000)

    # Готовим скейлер данных для обучения
    observation_examples = np.array([env.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # размерность входных данных
    input_size = len(observation_examples[0])

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(scaler, input_size, learning_rate=0.001)
    value_estimator = ValueEstimator(scaler, input_size, learning_rate=0.1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        results = actor_critic(env, policy_estimator, value_estimator, discount_factor=0.98, length=5000)