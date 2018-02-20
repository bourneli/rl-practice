# -*- coding:utf-8 -*-

import numpy as np
from lib.common import make_epsilon_greedy_policy
from lib.common import EpisodeStats
import sys


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, render=False):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
        render: show the process
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # # Print out which episode we're on, useful for debugging.
        # # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end='')
        sys.stdout.flush()

        state = env.reset()
        while True:
            if render:
                env.render()

            # on-policy epsilon贪心策略
            action = np.random.choice(np.arange(env.action_space.n), p=policy(state))
            next_state, reward, done, _ = env.step(action)

            # off-policy 贪心策略
            max_reward = max(estimator.predict(next_state))

            # 学习拟合函数
            target_y = reward + discount_factor * max_reward
            estimator.update(state, action, target_y)

            # 更新状态
            state = next_state
            stats.episode_lengths[i_episode] += 1
            stats.episode_rewards[i_episode] += reward

            if done:
                break

    return stats
