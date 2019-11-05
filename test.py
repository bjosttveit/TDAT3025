import gym
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import numpy as np
from tensorflow_core.python.keras.optimizers import Adam

env = gym.make("CarRacing-v0").env
ENV_NAME="CarRacing-v0"

print(env.action_space)
print(env.observation_space)
print(env.observation_space.shape[2])
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQN:
   def __init__(self, observation_space, action_space):
       self.exploration_rate = EXPLORATION_MAX
       self.model = models.Sequential()
       self.model.add(layers.Conv2D(96, (3, 3), activation='relu', input_shape=(96, 96, 3))) # (96,96,3)
       self.model.add(layers.MaxPooling2D((2, 2)))
       self.model.add(layers.Conv2D(192, (3, 3), activation='relu'))
       self.model.add(layers.MaxPooling2D((2, 2)))
       self.model.add(layers.Conv2D(192, (3, 3), activation='relu'))
       self.model.add(layers.Flatten())
       self.model.add(layers.Dense(64, activation='relu'))
       self.model.add(layers.Dense(10, activation='softmax'))
       self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', lr=LEARNING_RATE) #, metrics=['accuracy']
       self.model.summary()

   def remember(self, state, action, reward, next_state, done):
       self.memory.append((state, action, reward, next_state, done))

   def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.randomState()
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

   def randomState(self):
       return np.array([random.uniform(-1,1),random.uniform(0,1), random.uniform(0,1)])

   def experience_replay(self):
       if len(self.memory) < BATCH_SIZE:
           return
       batch = random.sample(self.memory, BATCH_SIZE)
       for state, action, reward, state_next, done in batch:
           q_update = reward
           if not done:
               q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
           q_values = self.model.predict(state)
           q_values[0][action] = q_update
           self.model.fit(state, q_values, verbose=0)
       self.exploration_rate *= EXPLORATION_DECAY
       self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

#env.observation_space
#dqn = DQN(env.observation_space, env.action_space)
def cartpole():
    env = gym.make(ENV_NAME)
    #score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space
    dqn_solver = DQN(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        print(state)
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                #score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

cartpole()

