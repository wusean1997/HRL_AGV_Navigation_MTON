#!/usr/bin/env python
# coding:utf-8
# Authors: Sean #
from __future__ import division
import pickle
import rospy
import os
import json
import numpy as np
import random
import time
import sys
from collections import deque
from std_msgs.msg import Float32, Float32MultiArray
from environment import Env
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gc
import torch.nn as nn
import math
import matplotlib.pyplot as plt
# import visdom
import json
#import gym
from model import Actor, Critic
# from tensorboardX import SummaryWriter


def train_model(actor, critic, target_critic, mini_batch, 
                actor_optimizer, critic_optimizer, alpha_optimizer,
                target_entropy, log_alpha, alpha):

    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    next_states = np.vstack(mini_batch[:, 3])
    dones = list(mini_batch[:, 4])


    if args.is_using_gpu:
        states        = torch.FloatTensor(states).cuda(args.gpu_device)
        next_states   = torch.FloatTensor(next_states).cuda(args.gpu_device)
        actions       = torch.FloatTensor(actions).squeeze(1).cuda(args.gpu_device)
        rewards       = torch.FloatTensor(rewards).unsqueeze(1).cuda(args.gpu_device)
        dones         = torch.FloatTensor(dones).unsqueeze(1).cuda(args.gpu_device)
    else:
        states        = torch.FloatTensor(states)
        next_states   = torch.FloatTensor(next_states)
        actions       = torch.FloatTensor(actions).squeeze(1)
        rewards       = torch.FloatTensor(rewards).unsqueeze(1)
        dones         = torch.FloatTensor(dones)

    """ Alpha Loss """
    # re-sample the action
    mu, std = actor(states)
    policy, log_policy = eval_action(mu, std)

    alpha_loss = -(log_alpha * (log_policy.cpu() + target_entropy).detach()).mean() # J(α) = Est∼D,at∼πt[−αlogπt(at|st)−αH^~]
    """ Update Alpha """
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha = torch.exp(log_alpha) 
    alpha_tlogs = alpha.clone() # For TensorboardX logs
    
    # ---- loss of the actor ---- #
    """ Actor Loss """
    q_value1, q_value2 = critic(states, policy)
    min_q_value = torch.min(q_value1, q_value2)
    actor_loss = ((alpha * log_policy.cpu()) - min_q_value.cpu()).mean() # Jπ(ϕ) = 𝔼st∼D,εt∼N[α * logπϕ(fϕ(εt;st)|st) − Qθ(St,fϕ(εt;st))]
    
    
    """ Update Parameters """
    """ Critic Loss """
    criterion = torch.nn.MSELoss()

    # Use two Q-functions to improve performance by reducing overestimation bias.
    q_value1_pred, q_value2_pred = critic(states, actions)
    
    # get target (Training Q function)
    # with torch.no_grad():
    mu, std = actor(next_states)
    next_policy, next_log_policy = eval_action(mu, std) # new_action, log_policy
    target_next_q_value1, target_next_q_value2 = target_critic(next_states, next_policy)
    # Take the min of the two Q-Values (Double-Q Learning)
    min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
    # V(St)=𝔼at∼π[Q(St,at)−αlogπ(at|st)]
    min_target_next_q_value = min_target_next_q_value.squeeze(1).cpu() - alpha * next_log_policy.squeeze(1).cpu() 
    if args.is_using_gpu:
        target = rewards + (1-dones) * args.gamma * min_target_next_q_value.cuda(args.gpu_device) # next_q_value
    else:
        target = rewards + (1-dones) * args.gamma * min_target_next_q_value # next_q_value
        
    # ---- loss of critics ---- #
    critic_loss1 = criterion(q_value1_pred.squeeze(1), target.detach()) # JQ(θ) = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[Vθ^¯(st+1)]))^2]
    critic_loss2 = criterion(q_value2_pred.squeeze(1), target.detach()) # JQ(θ) = 𝔼(st,at)~D[0.5(Q2(st,at) - r(st,at) - γ(𝔼st+1~p[Vθ^¯(st+1)]))^2]
    
    """
        Update networks
    """
    """ Update Critic """
    # update Q1
    critic_optimizer.zero_grad() # 0 gradient
    critic_loss1.backward()      # Backpropagation
    critic_optimizer.step()      # Update the parameters of generate network
    # update Q2
    critic_optimizer.zero_grad() # 0 gradient
    critic_loss2.backward()      # Backpropagation
    critic_optimizer.step()      # Update the parameters of generate network
    
    """ Update Actor """
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    """
    Soft Updates
    """
    soft_target_update(critic, target_critic, args.tau)
    
    # return alpha
    return critic_loss1.item(), critic_loss2.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

def main():
    # Initialize the Environment
    env = Env()
    start_time = time.time()
    past_action = np.array([0.,0.])

    state_size = args.state_size
    action_size = args.action_size
    print('State Dimensions:', state_size)
    print('Action Dimensions:', action_size)
    print('Velocity min: ' + str(args.action_v_Min) + ' m/s and ' + str(args.action_w_Min) + ' rad/s')
    print('Velocity Max: ' + str(args.action_v_Max) + ' m/s and ' + str(args.action_w_Max) + ' rad/s')
    

    # Actor & Critic network & Target Critic network
    if args.is_using_gpu:
        actor = Actor(state_size, action_size, args).cuda(args.gpu_device)
        critic = Critic(state_size, action_size, args).cuda(args.gpu_device)
        target_critic = Critic(state_size, action_size, args).cuda(args.gpu_device)
    else:
        actor         = Actor(state_size, action_size, args)
        critic        = Critic(state_size, action_size, args)
        target_critic = Critic(state_size, action_size, args)

    # Actor & Critic optimizer
    actor_optimizer  = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    # Hard target updete (copy parameters to the target network)
    hard_target_update(critic, target_critic)

    # initialize automatic entropy tuning
    # Target Entropy : −dim(A), (-|A|) (e.g. , -6 for HalfCheetah-v2) as given in the paper
    if args.is_using_gpu:
        target_entropy = -torch.prod(torch.FloatTensor(action_size).cuda(args.gpu_device)).item()
    else:
        target_entropy = -torch.prod(torch.FloatTensor(action_size)).item()
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha = torch.exp(log_alpha)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    # TensorboardX
    # writer = SummaryWriter(args.logdir)

    # Visdom
    # vis = visdom.Visdom(env=args.plot_env)

    # Initialize the Replay buffer
    replay_buffer = deque(maxlen=args.replaybuffer_size)
    
    #Load Replay_buffer
    dirname = "20210928-16-31_stage1"
    load_episode= 300
    replay_buffer = pickle.load(open("/home/eric/catkin_ws/src/hrl_project/base_action/src/SAC_model/" + dirname + "/" + "save.data", "rb"))
    
    #Load ac-model
    actor, critic, target_critic = load_models(actor, critic, target_critic, load_episode, dirname)
    
    rewards = []
    
    # update counts
    # updates = 0
    
    # Training Loop
    initGoal = True
    for episode in range(args.start_episodes, args.max_episodes):
        # done
        done = False
        # rewards
        episode_reward = 0
        # initial state
        state = env.reset(initGoal)
        initGoal = False
        state = np.reshape(state, [1, state_size])

        for step in range(args.max_steps):      
            state = np.float32(state)
            # take the action from the policy
            mu, std = actor(torch.FloatTensor(state).cuda()) 
            action = get_action(mu, std)
            #action[0] = round(action[0], 1)
            #action[1] = round(action[1], 1)

            unnorm_action = np.array([action_unnormalized(action[0], args.action_v_Max, args.action_v_Min), action_unnormalized(action[1], args.action_w_Max, args.action_w_Min)])
            next_state, reward, done = env.step(unnorm_action, past_action)
            
            past_action = unnorm_action
            episode_reward += round(reward,3)
            next_state = np.reshape(next_state, [1, state_size])
            

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print("state:", state)
            print('raw_action: ', action)
            print('action: ', unnorm_action)
            print("\033[1;32mEpisode: %d Step: %d Reward: %.3f Time: %d:%02d:%02d "% (episode, step, reward, h, m, s)+ "\033[0m")
            print("----------------------------------------------------------------------------------------------------")
            
            next_state = np.float32(next_state)

            # store in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
                
            if len(replay_buffer) > args.replaybuffer_size:
                replay_buffer.popleft()
                
            state = next_state
            

            if len(replay_buffer) > args.batch_size and args.is_training:
                # sample the batch from memory
                mini_batch = random.sample(replay_buffer, args.batch_size)
                
                actor.train(), critic.train(), target_critic.train() # Enable BatchNormalization & Dropout
                critic_loss1, critic_loss2, actor_loss, alpha_loss, alpha =\
                    train_model(actor, critic, target_critic, mini_batch, 
                                actor_optimizer, critic_optimizer, alpha_optimizer,
                                target_entropy, log_alpha, alpha)
                
                
                # writer.add_scalar('loss/critic_1', critic_loss1, updates)
                # writer.add_scalar('loss/critic_2', critic_loss2, updates)
                # writer.add_scalar('loss/policy', actor_loss, updates)
                # writer.add_scalar('loss/entropy_loss', alpha_loss, updates)
                # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                # vis.line(X=np.array([updates]), Y=np.array([critic_loss1]), win='critic_loss1_win', update='append', name='Critic_loss1', opts=dict(title='Critic_loss1', xlabel='step', ylabel= 'loss', showlegend=True))
                # vis.line(X=np.array([updates]), Y=np.array([critic_loss2]), win='critic_loss2_win', update='append', name='Critic_loss2', opts=dict(title='Critic_loss2', xlabel='step', ylabel= 'loss', showlegend=True))
                # vis.line(X=np.array([updates]), Y=np.array([actor_loss]), win='policy_loss_win', update='append', name='policy_loss', opts=dict(title='Policy loss', xlabel='step', ylabel= 'loss', showlegend=True))
                # vis.line(X=np.array([updates]), Y=np.array([alpha_loss]), win='entropy_loss_win', update='append', name='entropy_loss', opts=dict(title='Entropy loss', xlabel='step', ylabel= 'loss', showlegend=True))
                # vis.line(X=np.array([updates]), Y=np.array([alpha]), win='alpha_win', update='append', name='alpha', opts=dict(title='Entropy temprature', xlabel='step', ylabel= 'alpha', showlegend=True))
                # updates +=1               
                

            # elif not args.is_training:
            #     actor.eval(), critic.eval(), target_critic.eval() # Disable BatchNormalization & Dropout

            #if done:
                #break


        print("\033[1;31mReward per episode: " + str(episode_reward) + "\033[0m" )

        rewards.append(episode_reward)

        # writer.add_scalar('Reward', episode_reward, episode)

        # Using Visdom draw reward & success rate
        # vis.line(X=np.array([episode]), Y=np.array([episode_reward]), win='reward_win', update='append', name='Reward', opts=dict(title=args.plot_title, xlabel='Episode', ylabel= 'Reward', showlegend=True))
        # vis.line(X=np.array([episode]), Y=np.array([np.round(np.mean(rewards[-100:]),2)]), win='Average reward_win', update='append', name='Average Reward', opts=dict(title=args.plot_title, xlabel='Episode', ylabel= 'Reward', showlegend=True))
        # vis.line(X=np.array([episode]), Y=np.array([episode_reward]), win='test_win', update='append', name='Reward', opts=dict(title=args.plot_title, xlabel='Episode', ylabel= 'Reward', showlegend=True))
        # vis.line(X=np.array([episode]), Y=np.array([np.round(np.mean(rewards[-100:]),2)]), win='test_win', update='append', name='Average Reward')
        
        # if episode < 100:
        #     vis.line(X=np.array([episode]), Y=np.array([0]), win='SuccessSmallRate_win', update='append', name='Success Rate', opts=dict(title=args.plot_title+' (Normal)', xlabel='Episode', ylabel= 'Success rate(%)', showlegend=True))
        #     vis.line(X=np.array([episode]), Y=np.array([0]), win='SuccessBigRate_win', update='append', name='Success Rate', opts=dict(title=args.plot_title+' (Hard)', xlabel='Episode', ylabel= 'Success rate(%)', showlegend=True))
        # else :
        #     vis.line(X=np.array([episode]), Y=np.array([sum(successesSmall[-100:])]), win='SuccessSmallRate_win', update='append', name='Success Rate')
        #     vis.line(X=np.array([episode]), Y=np.array([sum(successesBig[-100:])]), win='SuccessBigRate_win', update='append', name='Success Rate')

        # Using Comet_ml draw reward & success rate
        # experiment.log_metric("Reward",episode_reward,step=episode)
        # experiment.log_metric("Average Reward",np.round(np.mean(rewards[-100:]),2),step=episode)
        # experiment.log_metric("Success Rate (Normal)",np.round(np.mean(rewards[-100:]),2),step=episode)
        # experiment.log_metric("Success Rate (Hard)",np.round(np.mean(rewards[-100:]),2),step=episode)

        # Using TensorboardX draw reward & success rate
        # writer.add_scalar('log/score', float(episode_reward), episode)

        # Release
        gc.collect()

        # Save log
        if episode % args.save_treshold == 0:
            with open(args.save_path + 'reward.json', 'wb') as f:
                json.dump(rewards, f)
                # f.write('\n')
            f.close()
            # rewards = []

            with open(args.save_path + 'save.data', 'wb') as f_rb:
                pickle.dump(replay_buffer, f_rb)
            f_rb.close()

            save_models(actor, critic, target_critic, episode)


if __name__ == '__main__':
    rospy.init_node('Sac', argv=sys.argv, anonymous=False, disable_signals=True, log_level=rospy.INFO)
    sys.argv = rospy.myargv(argv=sys.argv)
    import config
    from utils import *
    args = config.args
    #--------------------------Check Directory Path----------------------------------#
    dirPath = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    # -----------------------------Comet_ml Setting-----------------------------------#
    #Create an experiment
    # experiment = Experiment(api_key = args.api_key,
    #                         project_name = args.project_name, workspace=args.workspace)
    # --------------------------------------------------------------------------------#
    main()
