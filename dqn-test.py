from collections import deque
import random

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TokyoDrifter(keras.Model):
    def __init__(self, **kwargs):
        super(TokyoDrifter, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(256, (4, 4), activation='relu', input_shape=(96,96,3))
        self.pool1 = layers.MaxPool2D(2,2)
        self.conv2 = layers.Conv2D(256, (4, 4), activation='relu')
        self.pool2 = layers.MaxPool2D(2,2)
        self.conv3 = layers.Conv2D(256, (4, 4), activation='relu')
        self.pool3 = layers.MaxPool2D(2,2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(288, activation='relu')
        self.dense2 = layers.Dense(3, activation='linear')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

class DQN:
    def __init__(self, model, env, gamma=0.85, mem_size=1000000, batch_size=64, expl_max=1.0, expl_min=0.01, expl_decay=0.995):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=mem_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.expl_max = expl_max
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max

    def train(self, epochs):
        for e in range(epochs):
            s = self.env.reset()
            s = np.array(s[None, :], dtype=np.float16)
            d = False
            while not d:
                a = self.action(s)
                sn, r, d, _ = env.step(a)
                sn = np.array(sn[None, :], dtype=np.float16)
                self.remember(s, a, r, sn, d)
                s = sn
                self.experience_replay()

    
    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for s, a, r, sn, d in batch:
            q_update = r
            if not d:
                q_update = (r + self.gamma * np.amax(self.model.predict(sn)[0]))
            q_values = self.model.predict(s)
            q_values[0][a] = q_update
            self.model.fit(s, q_values, verbose=0)
        self.exploration_rate *= self.expl_decay
        self.exploration_rate = max(self.expl_min, self.exploration_rate)

    def remember(self, s, a, r, sn, d):
        self.memory.append((s, a, r, sn, d))

    def action(self, state):
        if np.random.rand() < self.expl_rate:
            return [np.random.uniform(-1,1), np.random.uniform(0,1), np.random.uniform(0,1)]
        a = self.model(state)
        return a




model = TokyoDrifter()
model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
env = gym.make("CarRacing-v0")

dqn = DQN(model, env)

dqn.train(10)