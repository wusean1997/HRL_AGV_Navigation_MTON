#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #
import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
#from std_msgs.msg import Float32MultiArray
from environment import Env
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
#SAC#
import pickle
from std_msgs.msg import Float32, Float32MultiArray
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gc
import torch.nn as nn
import math
from model import Actor
from utils import *


EPISODES = 6000

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('area_dqn/src', 'area_dqn/src/save_model/')
        self.result = Float32MultiArray()

        self.load_model = True
        self.load_episode = 2950
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 500
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == '__main__':
    rospy.init_node('turtlebot3_dqn_stage_1')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    history_action = Float32MultiArray()

    state_size = 26
    action_size = 9

    env = Env(action_size)

    #Basic SAC#####################################################################################################################################
    base_state_size = 26
    base_action_size = 2
    hidden_size = 256
    gpu_device = 0
    max_action_steps = 25
    action_v_Max = 0.52
    action_v_Min = 0.0
    action_w_Max = 2.79
    action_w_Min = -2.79
    action_actor = Actor(base_state_size, base_action_size, hidden_size).cuda(gpu_device)
    dirname = "basic_laser24_2" 
    load_episode= 2200
    action_actor = load_action_models(action_actor, load_episode, dirname)
    ################################################################################################################################################
    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    initialGoal = True
    initSubGoal = True

    for e in range(agent.load_episode + 1, EPISODES):
        get = False
        getgoal = False
        done = False
        fin_done = False
        state, goal_x, goal_y = env.reset()
        print("\033[1;35mEpisode" + str(e) + "| Goal[" + str(goal_x) + "," + str(goal_y) +"]\033[0m" )
        print("---------------------------------------")
        score = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)
            #Action########################################################################

            action_state, subgoal_x, subgoal_y = env.action_reset(action)
            action_state = np.reshape(action_state, [1, base_state_size])
            #print("Subgoal:[" + str(subgoal_x) + "," + str(subgoal_y) + "]")
            
            for action_step in range(max_action_steps): 
                action_state = np.float32(action_state)
                action_mu, action_std = action_actor(torch.FloatTensor(action_state).cuda())
                base_action = get_action(action_mu, action_std)

                unnorm_base_action = np.array([action_unnormalized(base_action[0], action_v_Max, action_v_Min), action_unnormalized(base_action[1], action_w_Max, action_w_Min)])
                next_action_state, done, get, getgoal = env.action_step(unnorm_base_action)
                action_state = next_action_state

                next_action_state = np.reshape(next_action_state, [1, base_state_size])
                next_state = np.float32(next_action_state)
                action_state = next_action_state

                if done or get or getgoal:
                    break
            ####################################################################################
            next_state, reward = env.step(done)
            print("\033[1;34mStep" + str(t) + "| Action: " + str(action) + "| Subgoal[" + str(subgoal_x) + "," + str(subgoal_y) + "] | Reward: " + str(reward) +"\033[0m" )
            print("---------------------------------------")

            agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            history_action.data = [action, score, reward]
            pub_get_action.publish(history_action)

            if e % 50 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done or getgoal:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                print("\033[1;36mEpisode" + str(e)+ " |score:" + str(score) + " |memory:" + str(len(agent.memory)) + " |epsilon:" + str(agent.epsilon) + " |time:" + str(h) + "h" + str(m) + "m" + str(s) + "s\033[0m")
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
