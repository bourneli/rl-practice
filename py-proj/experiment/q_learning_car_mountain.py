# -*- coding:utf-8 -*-

import csv
import gym
from lib.q_learning import q_learning
from lib.common import ActionValueEstimator
from lib.features import RBFFeatureizer, PassFeaturizer, ScaleFeaturizer
import os


env = gym.make('MountainCar-v0')

num_episodes = 200
discount_factor = 1
epsilon = 0
epsilon_decay = 1
render = False

# 特征透传
estimator_pass = ActionValueEstimator(env, PassFeaturizer())
stats_pass = q_learning(env, estimator_pass, num_episodes, discount_factor, epsilon, epsilon_decay, render=render)

# 标准化特征
estimator_scale = ActionValueEstimator(env, ScaleFeaturizer(env))
stats_scale = q_learning(env, estimator_scale, num_episodes, discount_factor, epsilon, epsilon_decay, render=render)

# RBF特征提取
estimator_RBF = ActionValueEstimator(env, RBFFeatureizer(env))
stats_RBF = q_learning(env, estimator_RBF, num_episodes, discount_factor, epsilon, epsilon_decay, render=render)

# 写结果
current_path = os.path.abspath(__file__)
data_path = os.path.abspath(current_path + '/../../../data/')
result_file = 'q_learning_car_mountain.csv'
with open(data_path+'\\'+result_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(('feature', 'episode', 'length', 'reward'))    # field header

    w.writerows([('Pass', i, stats_pass.episode_lengths[i], stats_pass.episode_rewards[i])
                 for i in range(0, num_episodes)])
    w.writerows([('Scale', i, stats_scale.episode_lengths[i], stats_scale.episode_rewards[i])
                 for i in range(0, num_episodes)])
    w.writerows([('RBF', i, stats_RBF.episode_lengths[i], stats_RBF.episode_rewards[i])
                 for i in range(0, num_episodes)])

print("\nComplete")
