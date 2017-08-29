import collections
import itertools
import os
import random

import tensorflow as tf

import numpy as np
import pandas as pd
import pickle

import logging.config

from sklearn.model_selection import train_test_split

from rl.actor import Actor
from rl.critic import Critic

logging.config.fileConfig('logging.conf')

logger = logging.getLogger('default')

from datautils import slice_array, load_returns
from distribution import MaximumEntropyPDF
from preprocessing import prices_data_file, models_data_file
from rl.tradingworld import TradingWorld

EpisodeStats = collections.namedtuple("Stats", ["episode_lengths", "episode_rewards"])

TradingResults = collections.namedtuple("TradingResults", ["strategy_points", "bnh_points", "mu", "sigma"])


CREATE_STATES = False
V_START = -0.007
V_END = 0.007
V_SIZE = 52
# MODELS_FILE = "data/.temp/models_EURUSD_s100_n40000_w12_log.csv"
STATES_FILE = "data/.temp/states_EURUSD_s100_n40000_w12_log.pcl"

INSTRUMENT = "EURUSD"
WINDOW = 12
START = 100
N = 40000
PRICE_FIELD = "bid_close"

EPISODE_LENGTH = 150 # hours

ACTION_SPACE_SIZE = 5
ACTION_SHIFT = int((ACTION_SPACE_SIZE - 1) / 2)

EXPLORATION = 0.1

EXPERIMENT_DIR="data/.temp/experiment"


def create_states(interval_start, interval_end, vectorization_size):
    models_file = models_data_file(INSTRUMENT, START, N, WINDOW)
    prices_file = prices_data_file("EURUSD")
    df = pd.DataFrame.from_csv(models_file, sep=";")
    models = MaximumEntropyPDF.from_df(df)
    states = []
    counter = 0
    for model in models:
        states.append(model.vectorized_cdf(interval_start, interval_end, vectorization_size))
        logger.debug("%d models processed" % (counter))
        counter += 1

    states = np.array(states)
    window = V_SIZE - 2
    states = np.array(list(states[i: (i + window)] for i in range(0, states.shape[0] - window - 1, 1)))
    points = load_returns(prices_file, PRICE_FIELD, START + window, N, rtype="points")
    pickle.dump((states, points), open(STATES_FILE, "wb"))
    return states, points

def create_episodes(states, points, length=EPISODE_LENGTH):
    episodes = []
    for i in range(0, states.shape[0] - length, length):
        episodes.append((states[i:i+length], points[i:i+length]))
    return episodes

def play_episodes(env, actor, critic, num_episodes, sess, discount_factor=1.0):
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
        state = env.reset()
        episode = []
        rewards = np.zeros(EPISODE_LENGTH)
        errors = np.zeros(EPISODE_LENGTH)
        values = np.zeros(EPISODE_LENGTH)
        # Одна прогонка обучения
        for t in itertools.count():
            # Делаем очередной шаг стратегии
            action = actor.choose_action(state, sess=sess)

            if random.random() < EXPLORATION:
                action = random.randint(0, ACTION_SPACE_SIZE - 1)

            trade = action - ACTION_SHIFT

            logger.debug("Value prediction is %f for %dth step" % (trade, t))
            next_state, reward, done = env.step(trade)
            # Сохраняем переход
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))
            # Обновляем статистику эпизода
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            rewards[t] = reward
            # Рассчитываем целевое TD значение
            value_next = critic.predict(next_state, sess=sess)
            td_target = reward + discount_factor * value_next
            td_error = td_target - critic.predict(state, sess=sess)
            errors[t] = td_error
            values[t] = value_next
            # Обновляем апроксиматор ценности состояния
            critic.update(state, td_target, sess=sess)
            # Обновляем апроксиматор политики
            actor.update(state, td_error, action, sess=sess)
            if done:
                break
            state = next_state

        logger.info("Episode {}/{} ({}), max/min/mean predicted value {}/{}/{}"
                    .format(i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1], np.max(values), np.min(values), np.mean(values)))
        # logger.info("Squared error id %f", np.sum(np.power(errors, 2)))

    return stats

if __name__ == '__main__':

    if CREATE_STATES:
        states, points = create_states(V_START, V_END, V_SIZE)
    else:
        states, points = pickle.load(open(STATES_FILE, "rb"))

    episodes = create_episodes(states, points)
    train, test = train_test_split(episodes, test_size=0.2)

    tf.set_random_seed(1234)
    sess = tf.Session()

    actor = Actor(V_SIZE, ACTION_SPACE_SIZE)
    critic = Critic(V_SIZE, ACTION_SPACE_SIZE)

    sess.run(tf.global_variables_initializer())

    # Создаем дириктории для чекпоинтов
    checkpoint_dir = os.path.join(EXPERIMENT_DIR, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(EXPERIMENT_DIR, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Загружаем чекпоинт, если есть
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    for i in range(25000):

        # Сохраняем текущее состояние модели
        saver.save(sess, checkpoint_path)

        ind = np.random.random_integers(0, len(train)-1)
        states, points = train[ind]
        states = np.nan_to_num(states)
        world = TradingWorld(states, points, max_allowed_position=3)
        stats = play_episodes(world, actor, critic, 4, sess)
        logger.debug("Mean reward %f" % np.mean(stats.episode_rewards))

