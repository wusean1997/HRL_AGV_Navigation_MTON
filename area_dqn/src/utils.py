# Authors: Eric
import torch
from torch.distributions import Normal
import numpy as np

#----------------------------------------
def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = round(np.clip(action, low, high), 3)
    return action

# https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
#**********************************

def get_action(mu, std): 
    """ Select the action. """
    # gaussian distribution
    normal = Normal(mu, std)

    # sample with reparameterization tricks
    z = normal.rsample().cuda(0) # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z)
    action = action.detach().cpu().numpy()
    return action[0]

def eval_action(mu, std, epsilon=1e-6):
    # gaussian distribution
    normal = Normal(mu, std)

    # sample with reparameterization tricks   
    z = normal.rsample().cuda(0) # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z)
    log_prob = normal.log_prob(z)

    # Enforcing Action Bounds
    log_prob -= torch.log(1 - action.pow(2) + epsilon) # log likelihood
    log_policy = log_prob.sum(1, keepdim=True) # sum through all actions

    return action, log_policy

def hard_target_update(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)
    # target_net.load_state_dict(net.state_dict())

def soft_target_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def load_action_models(actor, episode_count, dirname):#, critic, target_critic, episode_count, dirname):
    actor.load_state_dict(torch.load("/home/eric/catkin_ws/src/hrl_project/base_action/src/SAC_model/" + dirname + "/" + str(episode_count)+ '_actor.pth'))
    #critic.load_state_dict(torch.load("/home/eric/catkin_ws/src/hrl_project/base_action/src/SAC_model/" + dirname + "/" + str(episode_count)+ '_critic.pth'))
    #target_critic.load_state_dict(torch.load("/home/eric/catkin_ws/src/hrl_project/base_action/src/SAC_model/" + dirname + "/" + str(episode_count)+ '_target_critic.pth'))
    print("=====================================")
    print("........Action Model has been loaded........")
    print("=====================================")
    return actor#, critic, target_critic

