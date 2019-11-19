from collections import deque
import random
import time
import os
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations


class TokyoDrifter(keras.Model):
    def __init__(self, frames=5, **kwargs):
        super(TokyoDrifter, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(128, (5, 5))
        self.batchnorm1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(2, 2)
        self.conv2 = layers.Conv2D(128, (5, 5))
        self.batchnorm2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool2D(2, 2)
        self.conv3 = layers.Conv2D(256, (3, 3))
        self.batchnorm3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPool2D(2, 2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024)
        self.batchnorm4 = layers.BatchNormalization()
        self.dense2 = layers.Dense(4, activation='linear')
        
        self.build((None,48,42,frames))
    
    def call(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = activations.relu(x, alpha=0.01)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = activations.relu(x, alpha=0.01)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = activations.relu(x, alpha=0.01)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm4(x)
        x = activations.relu(x, alpha=0.01)
        return self.dense2(x)

class DQN:
    def __init__(self, model, env, gamma=0.95, optimizer=keras.optimizers.Adam(lr=0.0001), lossfunc=keras.losses.MSE, frames=5, exp_iterations=5, mem_size=128*1000, batch_size=128, expl_max=1.0, expl_min=0.01, expl_decay=0.95, save_interval=30):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=mem_size)
        self.weights = deque(maxlen=mem_size)
        self.gamma = gamma
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.exp_iterations = exp_iterations
        self.frames=frames
        self.save_interval = save_interval
        self.save_n = 0
        self.batch_size = batch_size
        self.expl_max = expl_max
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max
        self.steps = 0
        self.rewards = []
        self.explorerates = []
        self.avg_loss = []

    def action(self, q_values):
        if np.random.rand() < self.expl_rate:
            return random.randrange(0,4)
        return np.argmax(q_values)

    def drive(self, a):
        if a == 0: #Full gass
            return [0.0,0.2,0.0]
        elif a == 1: #Full gass og sving høyre
            return [1.0,0.2,0.0]
        elif a == 2: #Full gass og sving venstre
            return [-1.0,0.2,0.0]
        elif a == 3: #Full brems
            return [0.0,0.0,0.5]
    
    def experience_replay(self):
        #if self.steps < self.batch_size:
        if len(self.memory) < self.batch_size:    
            return
        print("Training on",self.batch_size,"samples for",self.exp_iterations,"epochs.","Current exploration rate",self.expl_rate)
        self.steps = 0
        
        avg_loss = 0
        for i in range(self.exp_iterations):

            self.update_weights()

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

            self.update_weights(batch)

        
        self.expl_rate = max(self.expl_min, self.expl_rate*self.expl_decay)
        self.avg_loss.append(avg_loss)
        tf.print("Average loss:",avg_loss)

    def update_weights(self, batch=None):
        if not batch:
            batch = random.sample(range(len(self.memory)), k=self.batch_size)
        x = np.array([self.memory[b][0] for b in batch])
        x2 = np.array([self.memory[b][4] for b in batch])

        modx = self.model(x)
        modx2 = self.model(x2)

        for i in range(len(batch)):
            v = self.memory[batch[i]][3]+np.max(modx2[i])*self.gamma if not self.memory[batch[i]][5] else self.memory[batch[i]][3]
            q_val = modx[i][self.memory[batch[i]][2]]
            self.weights[batch[i]] = np.absolute(q_val-v)
        
        #print(np.mean(self.weights),np.std(self.weights))


    def preprocess_image(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.image.crop_to_bounding_box(x, 0,0,84,96)
        x = tf.image.resize(x,(42,48),method="bilinear")
        x = tf.math.divide(x, 255)
        x = tf.math.subtract(1, x)
        #x = np.where(x<0.5,np.square(x), np.sqrt(x))*2-1
        
        #print(x)
        #plt.imshow(np.squeeze(x), cmap='gray')
        #plt.show()
        return x
    
    def stack_frames(self, state, new_state):
        states = tf.split(state[0],self.frames,-1)
        states.insert(0, new_state)
        states.pop()

        return tf.concat(states, -1)
    
    def train(self):
        try:
            self.starttime = time.time()
            self.name = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            os.mkdir(os.path.join("./", self.name))

            while True:
                episode_reward, steps_since_reward = 0,0
                
                self.env.reset()
                [self.env.step(self.drive(0)) for i in range(40)]
                
                s,_,_,_ = self.env.step(self.drive(0))
                s = self.preprocess_image(s)
                s = tf.concat([s]*self.frames, -1)
                s = np.array(s[None, :], dtype=np.float32)
                d = False
                episode_memory = []
                while not d:
                    self.steps += 1
                    #self.env.render()

                    q_values = self.model(s)
                    a = self.action(q_values)

                    sn, r, d, _ = self.env.step(self.drive(a))
                    episode_reward += r

                    r = np.clip(r, -1, 1)
                    
                    if r > 0:
                        steps_since_reward = 0
                    else:
                        steps_since_reward += 1
                    if steps_since_reward > 100:
                        d = True

                    sn = self.preprocess_image(sn)
                    sn = self.stack_frames(s, sn)
                    sn = np.array(sn[None, :], dtype=np.float32)

                    q2 = np.max(self.model(sn))
                    v = r+self.gamma*q2 if not d else r

                    self.weights.append(np.absolute(q_values[0][a]-v))

                    episode_memory.append([s[0],0,a,r,sn[0],d])

                    s = sn

                self.memory.extend(episode_memory)
                print("Reward",episode_reward)
                self.rewards.append(episode_reward)
                self.explorerates.append(self.expl_rate)
                self.experience_replay()

                if (time.time() - self.starttime) / 60 - self.save_interval*self.save_n > self.save_interval:
                    self.save()
                    self.save_n += 1

        except KeyboardInterrupt:
            self.save()

    def save(self):
        print("Saving model and plots...")

        elapsed = (time.time() - self.starttime)/3600
        timestring = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        self.model.save_weights(os.path.join("./",self.name,"model %s %.2f.h5" % (timestring, elapsed)))

        avg_rewards = pd.DataFrame(self.rewards).ewm(span=100).mean()
        plt.plot(self.rewards,'-', label="Actual", ms=2)
        plt.plot(avg_rewards,'-', label="100 episode EMA")
        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig(os.path.join("./",self.name,'reward %s %.2f.png' % (timestring, elapsed)))

        plt.clf()

        plt.plot(self.explorerates)
        plt.ylabel('Exploration rate')
        plt.xlabel('Episode')
        plt.savefig(os.path.join("./",self.name,'explore %s %.2f.png' % (timestring, elapsed)))

        plt.clf()

        plt.plot(self.avg_loss)
        plt.yscale("log")
        plt.ylabel('Average loss')
        plt.xlabel('Episode')
        plt.savefig(os.path.join("./",self.name,'loss %s %.2f.png' % (timestring, elapsed)))

        plt.clf()
    
    def test(self):
        try:
            while True:
                episode_reward, steps_since_reward = 0,0
                
                self.env.reset()
                [self.env.step(self.drive(0)) for i in range(40)]
                
                s,_,_,_ = self.env.step(self.drive(0))
                s = self.preprocess_image(s)
                s = tf.concat([s]*self.frames, -1)
                s = np.array(s[None, :], dtype=np.float32)
                d = False
                while not d:
                    self.steps += 1
                    self.env.render()

                    q_values = self.model(s)
                    a = np.argmax(q_values)

                    sn, r, d, _ = self.env.step(self.drive(a))
                    episode_reward += r
                    
                    if r > 0:
                        steps_since_reward = 0
                    else:
                        steps_since_reward += 1
                    if steps_since_reward > 100:
                        d = True

                    sn = self.preprocess_image(sn)
                    sn = self.stack_frames(s, sn)
                    sn = np.array(sn[None, :], dtype=np.float32)

                    s = sn

                print("Reward",episode_reward)
        except KeyboardInterrupt:
            pass


test = False

model = TokyoDrifter(frames=4)
env = gym.make("CarRacing-v0")


if test:
    model.load_weights("model bestrapperalivev3.h5")

dqn = DQN(model, env, expl_max=1.0, expl_min=0.01, expl_decay=0.995, gamma=0.95, exp_iterations=3, frames=4, save_interval=5)

if test:
    dqn.test()
else:
    dqn.train()
