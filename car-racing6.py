from collections import deque
import random
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class TokyoDrifter(keras.Model):
    def __init__(self, **kwargs):
        super(TokyoDrifter, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(128, (4, 4), activation='relu')
        self.pool1 = layers.MaxPool2D(2, 2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(7, activation='linear')
        
        self.build((None,24,21,3))
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

class DQN:
    def __init__(self, model, env, gamma=0.95, optimizer=keras.optimizers.Adagrad(lr=0.001), lossfunc=keras.losses.MSE, exp_iterations=5, mem_size=128*1000, batch_size=128, expl_max=1.0, expl_min=0.01, expl_decay=0.95):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=mem_size)
        self.weights = deque(maxlen=mem_size)
        self.gamma = gamma
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.exp_iterations = exp_iterations
        self.batch_size = batch_size
        self.expl_max = expl_max
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max
        self.steps = 0
        self.rndact = 0
        self.rndint = 0
        self.actint = 0

    def action(self, q_values):
        if self.actint > 0:
            self.actint += 1
            if self.actint > 2: self.actint = 0
            return np.argmax(q_values)
        
        if self.rndint > 0:
            self.rndint += 1
            if self.rndint > 2: self.rndint = 0
            return self.rndact
        
        if np.random.rand() < self.expl_rate:
            self.rndint = 1
            m = 7 #I starten vil tilfeldig valg være begrenset
            if self.expl_rate > 0.9: m = 2
            elif self.expl_rate > 0.5: m = 4
            elif self.expl_rate > 0.1: m = 6
            self.rndact = random.randrange(1,m) #tilfeldig er aldri å gjøre ingenting
            return self.rndact
        else:
            self.actint = 1
        return np.argmax(q_values)

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
            return [0.0,0.0,0.5]
        else: #a==0 Gjør ingenting
            return [0.0,0.0,0.0]
    
    def experience_replay(self):
        #if self.steps < self.batch_size:
        if len(self.memory) < self.batch_size:    
            return
        print("Training on",self.batch_size,"samples for",self.exp_iterations,"epochs.","Current exploration rate",self.expl_rate)
        self.steps = 0
        
        avg_loss = 0
        for i in range(self.exp_iterations):
            batch = random.choices(range(len(self.memory)), k=self.batch_size, weights=self.weights)
            x = np.array([self.memory[b][0] for b in batch])
            x2 = np.array([self.memory[b][4] for b in batch])
            modx2 = self.model(x2)

            with tf.GradientTape() as t:
                modx = self.model(x)
                y = []
                for j in range(len(batch)):
                    q_update = modx[j].numpy()
                    q_update[self.memory[batch[j]][2]] = self.memory[batch[j]][3] + np.max(modx2[j])*self.gamma if not self.memory[batch[j]][5] else self.memory[batch[j]][3]
                    y.append(q_update)
                current_loss = self.lossfunc(modx, y)
            grads = t.gradient(current_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            avg_loss += tf.reduce_mean(current_loss)/self.exp_iterations

            self.update_weights()

            modx = self.model(x)
            modx2 = self.model(x2)
            for j in range(len(batch)):
                v = self.memory[batch[j]][3]+np.max(modx2[j])*self.gamma if not self.memory[batch[j]][5] else self.memory[batch[j]][3]
                q_val = modx[j][self.memory[batch[j]][2]]
                self.weights[batch[j]] = np.power(q_val-v,2)

        
        self.expl_rate = max(self.expl_min, self.expl_rate*self.expl_decay)
        tf.print("Average loss:",avg_loss)

    def update_weights(self):
        batch = random.sample(range(len(self.memory)), k=self.batch_size)
        x = np.array([self.memory[b][0] for b in batch])
        x2 = np.array([self.memory[b][4] for b in batch])

        modx = self.model(x)
        modx2 = self.model(x2)

        for i in range(len(batch)):
            v = self.memory[batch[i]][3]+np.max(modx2[i])*self.gamma if not self.memory[batch[i]][5] else self.memory[batch[i]][3]
            q_val = modx[i][self.memory[batch[i]][2]]
            self.weights[batch[i]] = np.absolute(q_val-v)
        
        print(np.mean(self.weights),np.std(self.weights))


    def preprocess_image(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.image.resize(x,(24,24),method="bilinear")
        x = tf.image.crop_to_bounding_box(x, 0,0,21,24)
        x = tf.math.divide(x, 255)
        x = tf.math.subtract(1, x)
        x = np.where(x<0.5,np.square(x), np.sqrt(x))*2-1
        
        #print(x)
        #plt.imshow(np.squeeze(x), cmap='gray')
        #plt.show()
        return x
    
    def train(self):
        try:
            while True:
                episode_reward, steps_since_reward, self.rndint = 0,0,0
                
                self.env.reset()
                [self.env.step(self.drive(0)) for i in range(40)]
                
                s,_,_,_ = self.env.step(self.drive(0))
                s = self.preprocess_image(s)
                s = tf.concat((s,s,s), -1)
                s = np.array(s[None, :], dtype=np.float32)
                d = False
                episode_memory = [[],[],[],[],[],[]]
                while not d:
                    self.steps += 1
                    self.env.render()

                    q_values = self.model(s)
                    a = self.action(q_values)

                    sn, r, d, _ = self.env.step(self.drive(a))
                    episode_reward += r
                    
                    if r > 0:
                        steps_since_reward = 0
                    else:
                        steps_since_reward += 1
                    if steps_since_reward > 20:
                        d = True
                        #r -= 30

                    sn = self.preprocess_image(sn)
                    sn = tf.concat((sn,tf.split(s,3,-1)[0][0],tf.split(s,3,-1)[1][0]),-1)
                    sn = np.array(sn[None, :], dtype=np.float32)

                    q2 = np.max(self.model(sn))
                    v = r+self.gamma*q2 if not d else r

                    self.weights.append(np.absolute(q_values[0][a]-v))

                    episode_memory[0].append(s[0])
                    episode_memory[1].append(0)
                    episode_memory[2].append(a)
                    episode_memory[3].append(r)
                    episode_memory[4].append(sn[0])
                    episode_memory[5].append(d)

                    s = sn

                self.memory.extend(np.transpose(episode_memory))
                print("Reward",episode_reward)
                self.experience_replay()
        except KeyboardInterrupt:
            self.model.save_weights("model5.h5")


model = TokyoDrifter()

try:
    model.load_weights("model5.h5")
except OSError:
    pass

#model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
env = gym.make("CarRacing-v0")

dqn = DQN(model, env, expl_max=0.2, expl_min=0.01, expl_decay=0.995, gamma=0.99, exp_iterations=3)

dqn.train()
