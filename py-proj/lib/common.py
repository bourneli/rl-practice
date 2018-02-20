# -*- coding:utf-8 -*-

from sklearn.linear_model import SGDRegressor
import numpy as np
from collections import namedtuple

# 统计每个episode的长度与奖励
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class ActionValueEstimator:
    """
    动作值函数估计对象
    """

    def __init__(self, env, featurizer):

        self.featurizer = featurizer

        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.env = env
        self.models = []
        for _ in range(self.env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(self.env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        return self.featurizer.transform(state)

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        feature = self.featurize_state(s)
        if a is not None:
            return self.models[a].predict([feature])[0]
        else:
            return np.array([m.predict([feature])[0] for m in self.models])

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        self.models[a].partial_fit([self.featurize_state(s)], [y])
        return None


def make_epsilon_greedy_policy(estimator, epsilon, action_number):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        action_number: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        actions = np.ones(action_number, dtype=float) * epsilon / action_number
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        actions[best_action] += (1.0 - epsilon)
        return actions

    return policy_fn
