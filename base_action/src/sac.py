#!/usr/bin/env python
# Authors: Morrison #
from __future__ import division
import pickle
from comet_ml import Experiment
import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32, Float32MultiArray
from social_environment_camera import Env
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gc
import torch.nn as nn
import math
from collections import deque
import matplotlib.pyplot as plt
import visdom
import json
from model import ValueNetwork, SoftQNetwork, PolicyNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """ Store new transition. """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """ Sample randomly. """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)




def soft_q_update(batch_size, replay_buffer, value_net, target_value_net, soft_q_net, policy_net,
           value_optimizer, soft_q_optimizer, policy_optimizer,
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    if args.is_using_gpu:
        state      = torch.FloatTensor(state).cuda(1)
        next_state = torch.FloatTensor(next_state).cuda(1)
        action     = torch.FloatTensor(action).cuda(1)
        reward     = torch.FloatTensor(reward).cuda(1).unsqueeze(1)
        done       = torch.FloatTensor(np.float32(done)).cuda(1).unsqueeze(1)
    else:
        state      = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action     = torch.FloatTensor(action)
        reward     = torch.FloatTensor(reward).unsqueeze(1)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)
    #print('done', done)
    # criterion = nn.MSELoss()
    criterion = torch.nn.MSELoss()

    expected_q_value = soft_q_net(state, action)
    expected_value   = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = criterion(expected_q_value, next_q_value.detach())
    # Training Value Function
    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = criterion(expected_value, next_value.detach())
    # Training Policy Function
    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()


    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad() #0 gradient
    q_value_loss.backward() #Backpropagation
    soft_q_optimizer.step() #Update the parameters of generate network

    value_optimizer.zero_grad() #0 gradient
    value_loss.backward() #Backpropagation
    value_optimizer.step() #Update the parameters of generate network

    policy_optimizer.zero_grad() #0 gradient
    policy_loss.backward() #Backpropagation
    policy_optimizer.step() #Update the parameters of generate network
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

#----------------------------------------------------------
#-----------------------------------------------------
def save_models(policy_net, value_net, soft_q_net, episode_count):
    torch.save(policy_net.state_dict(), args.save_path + str(episode_count)+ '_policy_net.pth')
    torch.save(value_net.state_dict(), args.save_path + str(episode_count)+ 'value_net.pth')
    torch.save(soft_q_net.state_dict(), args.save_path + str(episode_count)+ 'soft_q_net.pth')
    print("====================================")
    print("Model has been saved...")
    print("====================================")

def load_models(policy_net, value_net, soft_q_net, episode_count):
    policy_net.load_state_dict(torch.load(args.save_path + str(episode)+ '_policy_net.pth'))
    value_net.load_state_dict(torch.load(args.save_path + str(episode)+ 'value_net.pth'))
    soft_q_net.load_state_dict(torch.load(args.save_path + str(episode)+ 'soft_q_net.pth'))
    print('***Models load***')
#****************************

    

# load_models(start_episodes)

# count = 0

#----------------------------------------
def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action
#**********************************
def main():
    # Initialize the Environment
    env = Env()
    start_time = time.time()
    past_action = np.array([0.,0.])

    action_dim = args.action_size
    state_dim  = args.state_size
    hidden_dim = args.hidden_size #512
    ACTION_V_MIN = args.action_v_Min # m/s
    ACTION_W_MIN = args.action_w_Min # rad/s
    ACTION_V_MAX = args.action_v_Max # m/s
    ACTION_W_MAX = args.action_w_Max # rad/s
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')

    # ValueNetwork & SoftQNetwork & PolicyNetwork
    value_net        = ValueNetwork(state_dim, hidden_dim).cuda(1)
    target_value_net = ValueNetwork(state_dim, hidden_dim).cuda(1)
    soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).cuda(1)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).cuda(1)

    

    # if is_using_gpu:
    #     value_criterion  = nn.MSELoss().cuda(1)
    #     soft_q_criterion = nn.MSELoss().cuda(1)
    # else:
    # value_criterion  = nn.MSELoss()
    # soft_q_criterion = nn.MSELoss()

    # Optimizer
    value_lr  = 3e-4
    soft_q_lr = 3e-4
    policy_lr = 3e-4
    value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
    soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    #Hard target updete
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)
    
    # Initialize the Replay buffer
    replay_buffer_size = 100000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    rewards     = []
    successesBig = []
    successesSmall = []

    

    # Visdom
    # vis = visdom.Visdom(env=args.plot_env)

    # Training Loop
    for episode in range(args.start_episodes, args.max_episodes):
        # done
        done = False
        # rewards
        rewards_current_episode = 0
        # initial state
        state = env.reset()

        for step in range(args.max_steps):
            state = np.float32(state)
            # take the action from the policy
            action = policy_net.get_action(state)

            unnorm_action = np.array([action_unnormalized(action[0], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)])
            
            if args.useLaser:
                # Laser
                next_state, reward, done = env.step(unnorm_action, past_action)
            else:
                # Social
                next_state, reward, done, successBig, successSmall = env.step(unnorm_action, past_action)
            
            
            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print('action: ', unnorm_action)
            if args.useLaser:
                print("\033[1;32mEpisode: %d Step: %d Reward: %.3f Time: %d:%02d:%02d "% (episode, step, reward, h, m, s)+ "\033[0m")
            else:
                print("\033[1;32mEpisode: %d Step: %d Reward: %.3f Time: %d:%02d:%02d Success Big: %d Success Small: %d"% (episode, step, reward, h, m, s, sum(successesBig), sum(successesSmall))+ "\033[0m")
            print("--------------------------------------------------------------------------")
            past_action = action

            rewards_current_episode += reward

            next_state = np.float32(next_state)

            # store in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > 2*args.batch_size and args.is_training:
                soft_q_update(args.batch_size, replay_buffer, value_net, target_value_net, soft_q_net, policy_net,
                              value_optimizer, soft_q_optimizer, policy_optimizer)
        
            state = next_state

            if done:
                break


        print("\033[1;31mReward per episode: " + str(rewards_current_episode) + "\033[0m" )
        # print('reward per ep: ' + str(rewards_current_episode))

        
        if args.useLaser:
            rewards.append(rewards_current_episode)
        else:
            successesBig.append(successBig)
            successesSmall.append(successSmall)
            rewards.append(rewards_current_episode)

        #Using Visdom draw reward
        # vis.line(X=np.array([episode]), Y=np.array([rewards_current_episode]), win='reward_win', update='append', name='Reward', opts=dict(title=args.plot_title, xlabel='Episode', ylabel= 'Reward', showlegend=True))
        # vis.line(X=np.array([episode]), Y=np.array([np.round(np.mean(rewards[-100:]),2)]), win='Average reward_win', update='append', name='Average Reward', opts=dict(title=args.plot_title, xlabel='Episode', ylabel= 'Reward', showlegend=True))
        # vis.line(X=np.array([episode]), Y=np.array([rewards_current_episode]), win='test_win', update='append', name='Reward', opts=dict(title=args.plot_title, xlabel='Episode', ylabel= 'Reward', showlegend=True))
        # vis.line(X=np.array([episode]), Y=np.array([np.round(np.mean(rewards[-100:]),2)]), win='test_win', update='append', name='Average Reward')
        

        # if episode < 100:
        #     vis.line(X=np.array([episode]), Y=np.array([0]), win='SuccessBigRate_win', update='append', name='Success Rate', opts=dict(title=args.plot_title+' (Hard)', xlabel='Episode', ylabel= 'Success rate(%)', showlegend=True))
        #     vis.line(X=np.array([episode]), Y=np.array([0]), win='SuccessSmallRate_win', update='append', name='Success Rate', opts=dict(title=args.plot_title+' (Normal)', xlabel='Episode', ylabel= 'Success rate(%)', showlegend=True))
            
        # else :
            
        #     vis.line(X=np.array([episode]), Y=np.array([sum(successesBig[-100:])]), win='SuccessBigRate_win', update='append', name='Success Rate')
        #     vis.line(X=np.array([episode]), Y=np.array([sum(successesSmall[-100:])]), win='SuccessSmallRate_win', update='append', name='Success Rate')
            
        #Using Visdom angle
            

        # experiment.log_metric("Reward",rewards_current_episode,step=episode)
        # experiment.log_metric("Average Reward",np.round(np.mean(rewards[-100:]),2),step=episode)
        
        
        # Release
        gc.collect()

        # Save log
        if episode % args.save_treshold == 0:
            with open(args.save_path + 'reward.json', 'wb') as f:
                json.dump(rewards, f)
                # f.write('\n')
            f.close()

            if args.useLaser:
                pass
            else:
                with open(args.save_path + 'successBig.json', 'wb') as f2:
                    json.dump(successesBig, f2)
                    # f2.write('\n')
                f2.close()

                with open(args.save_path + 'successSmall.json', 'wb') as f3:
                    json.dump(successesSmall, f3)
                    # f2.write('\n')
                f3.close()
            
            save_models(policy_net, value_net, soft_q_net, episode)
        
if __name__ == '__main__':
    # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    rospy.init_node('social_sac_stage_1_new', argv=sys.argv, anonymous=False, disable_signals=True, log_level=rospy.INFO)
    sys.argv = rospy.myargv(argv=sys.argv)
    import config
    # from utils import *
    args = config.args
    #--------------------------Check Directory Path----------------------------------#
    dirPath = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    #--------------------------------------------------------------------------------#
    #--comet_ml Setting---#
    #--Create an experiment---#
    # experiment = Experiment(api_key = args.api_key,
    #                         project_name = args.project_name, workspace=args.workspace)
    #--------------------------------------------------------------------------------#
    main()