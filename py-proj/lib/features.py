# -*- coding:utf8 -*-

import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


class Featurizer:

    def transform(self, data):
        pass


class PassFeaturizer(Featurizer):

    def transform(self, data):
        return data


class ScaleFeaturizer(Featurizer):

    def __init__(self, env):
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

    def transform(self, data):
        return self.scaler.transform([data])[0]


class RBFFeatureizer(Featurizer):

    def __init__(self, env):
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def transform(self, data):
        scaled = self.scaler.transform([data])
        return self.featurizer.transform(scaled)[0]


class Order2CrossFeaturizer(Featurizer):

    def transform(self, data):
        extended_size = int(data.size * (data.size - 1) / 2)
        extended_array = np.zeros(shape=extended_size)
        index = 0
        for i in range(0, data.size):
            for j in range(i + 1, data.size):
                extended_array[index] = data[i] * data[j]
                index += 1

        return np.concatenate((data, extended_array))
