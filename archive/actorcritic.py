import numpy as np
import pandas as pd

from archive.tradingworld import TradingWorld
from datautils import load_returns
from distribution import FpdFromHeatDiffusion


def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


def update_critic(utility_matrix, observation, new_observation, reward, alpha, gamma):
    u = utility_matrix[observation]
    u_t1 = utility_matrix[new_observation]
    delta = reward + gamma * u_t1 - u
    utility_matrix[observation] += alpha * (delta)
    return utility_matrix, delta


def update_actor(state_action_matrix, observation, action, delta, beta_matrix=None):
    if beta_matrix is None: beta = 1
    else: beta = 1 / beta_matrix[action,observation]
    state_action_matrix[action, observation] += beta * delta
    return state_action_matrix


def main():

    shift = 12
    start = 28045
    N = 5000
    points = load_returns("data/IB-EURUSD-8-XI-2016-2.csv", "ask_close", start + shift + 1, N, rtype="points") * 1e5

    df = pd.DataFrame.from_csv("../data/.temp/models_EUR_s28045_n800_w12_log_old.csv", sep=";")
    models = FpdFromHeatDiffusion.from_df(df)[:N]

    points = points[3000:]
    models = models[3000:]

    window_size = 1000
    moving_window = (0, window_size)
    step = 1

    model_features = pd.DataFrame.from_csv("data/.temp/models_features_EURUSD_s28045_n800_w12_log.csv")[:5000]
    env = TradingWorld(points, model_features, moving_window)

    state_action_matrix = np.random.random((3, env.size))
    utility_matrix = np.zeros(env.size)

    gamma = 0.99
    alpha = 0.001
    tot_epoch = 120

    rewards = np.zeros(N)
    cur_test_financial_position = 1

    for shift in range(env.lag(), 100, step):
        env.moving_window = (shift, shift + window_size)

        for epoch in range(tot_epoch):
            #Reset and return the first observation
            observation = env.reset()
            while True:
                #Estimating the action through Softmax
                action_array = state_action_matrix[:, observation]
                action_distribution = softmax(action_array)
                action = np.random.choice(3, 1, p=action_distribution)
                change_action = np.random.choice(2, 1, p=[0.99, 0.01])
                if change_action == 1:
                    action = np.random.choice(3, 1, p=[1.0/3, 1.0/3, 1.0/3])
                new_observation, reward, done = env.step(action)
                # придаем последним вознаграждениям больший вес
                wparam = 5 / (env.moving_window[1] - env.moving_window[0])
                weight = np.exp(-wparam*((env.moving_window[1] - env.moving_window[0]) - env.position - 1))
                reward = reward * weight
                utility_matrix, delta = update_critic(utility_matrix, observation,
                                                      new_observation, reward - 5, alpha, gamma)
                state_action_matrix = update_actor(state_action_matrix, observation,
                                                   action, delta, beta_matrix=None)

                observation = new_observation
                if done: break

        env.moving_window = (env.moving_window[0], env.moving_window[1] + step)
        env.finance_position = cur_test_financial_position
        observation = env.state()
        for forecast_pos in range(0, step):
            action_array = state_action_matrix[:, observation]
            action_distribution = softmax(action_array)
            action = np.random.choice(3, 1, p=action_distribution)
            # action = np.random.choice(3, 1, p=[1.0 / 3, 1.0 / 3, 1.0 / 3])
            new_observation, reward, _ = env.step(action)
            cur_test_financial_position = env.finance_position
            rewards[window_size + shift + forecast_pos] = reward
            print(np.sum(rewards))

    import matplotlib.pyplot as plt
    plt.plot(np.cumsum(rewards))
    plt.show()


            # if(epoch % print_epoch == 0):
            #     print("")
            #     print("Total reward after " + str(epoch+1) + " iterations:")
            #     print(total_reward)
            #
            #     import matplotlib.pyplot as plt
            #     plt.plot(rewards)
            #     plt.plot(np.cumsum(points))
            #     # plt.plot(means)
            #     # plt.plot(signal / 100)
            #     plt.show()


if __name__ == "__main__":
    main()