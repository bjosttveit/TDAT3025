from collections import deque
import random
import time

import tensorflow as tf
import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class TokyoDrifter(keras.Model):
    def __init__(self, **kwargs):
        super(TokyoDrifter, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(128, (4, 4), activation='relu', input_shape=(96,96,3))
        self.pool1 = layers.MaxPool2D(2,2)
        self.conv2 = layers.Conv2D(128, (4, 4), activation='relu')
        self.pool2 = layers.MaxPool2D(2,2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(9, activation='linear')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class DQN:
    def __init__(self, model, env, gamma=0.85, mem_size=128*100, batch_size=128, expl_max=1.0, expl_min=0.01, expl_decay=0.98):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=mem_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.expl_max = expl_max
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max
        self.steps = 0

    def train(self):
        try:
            while True:
                tot_r = 0
                tsr = 0# time since reward
                s = self.env.reset()
                s = np.array(s[None, :], dtype=np.float32)
                d = False
                while not d:
                    self.steps += 1
                    self.env.render()
                    a = self.action(s)
                    drive = self.drive(a)
                    sn, r, d, _ = self.env.step(drive)
                    tot_r += r
                    
                    if r > 0:
                        tsr = 0
                    else:
                        tsr += 1
                    if tsr > 20:
                        d = True

                    sn = np.array(sn[None, :], dtype=np.float32)
                    self.remember(s, a, r, sn, d)
                    s = sn
                    self.experience_replay()
                print("Reward",tot_r)
        except KeyboardInterrupt:
            self.model.save_weights("model.h5")
    
    def experience_replay(self):
        if len(self.memory) < 1280 or self.steps % self.batch_size != 0:
            return
        print("Training on",self.batch_size,"samples", "Current exploration rate:", self.expl_rate)
        #s1 = time.time()
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([elem[0] for elem in batch]).squeeze()
        actions = np.array([elem[1] for elem in batch])
        rewards = np.array([elem[2] for elem in batch])
        newstates = np.array([elem[3] for elem in batch]).squeeze()
        dones = np.array([elem[4] for elem in batch])

        predictions = self.model.predict(newstates)
        predictions = [np.amax(pred) for pred in predictions]

        q_update = np.where(dones, rewards, rewards + np.multiply(self.gamma,predictions))
        
        q_values = self.model.predict(states)
        for i in range(len(q_values)):
            q_values[i][actions[i]] = q_update[i]
        #print("preprocessing",time.time()-s1)
        
        #s2 = time.time()
        self.model.fit(states, q_values, batch_size=self.batch_size, verbose=0)
        #print("fit",time.time()-s2)
        
        self.expl_rate *= self.expl_decay
        self.expl_rate = max(self.expl_min, self.expl_rate)

    def remember(self, s, a, r, sn, d):
        self.memory.append((s, a, r, sn, d))

    def action(self, state):
        if np.random.rand() < self.expl_rate:
            m = 9
            if self.expl_rate > 0.5: m = 4 #Tilfeldig innebærer alltid å kjøre i starten slik at den lærer fort at den må bruke gassen for å komme fremover
            return random.randrange(0,m)
        return np.argmax(self.model(state))
        
    def drive(self, a):
        if a == 1: #Full gass
            return [0.0,1.0,0.0]
        elif a == 2: #Full gass og sving høyre
            return [1.0,1.0,0.0]
        elif a == 3: #Full gass og sving venstre
            return [-1.0,1.0,0.0]
        elif a == 4: #Sving høyre
            return [1.0,0.0,0.0]
        elif a == 5: #Sving venstre
            return [-1.0,0.0,0.0]
        elif a == 6: #Full brems
            return [0.0,0.0,1.0]
        elif a == 7: #Full brems og sving høyre
            return [1.0,0.0,1.0]
        elif a == 8: #Full brems og sving venstre
            return [-1.0,0.0,1.0]
        else: #a==0 Gjør ingenting
            return [0.0,0.0,0.0]


model = TokyoDrifter()
model.build((None,96,96,3))
try:
    model.load_weights("model.h5")
except OSError:
    pass
model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))
env = gym.make("CarRacing-v0")

dqn = DQN(model, env, expl_min=0.01, expl_max=0.9, expl_decay=0.95)

dqn.train()
