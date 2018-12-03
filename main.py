# At first, it only has a2c, in the next we will implement ppo
import copy
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arguments import get_args
from model import Policy

# import algo
from arguments import get_args
from visualize import visdom_plot

args = get_args()

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
  

def main():
  torch.set_num_threads(1)
  device = torch.device("cuda:0" if args.cuda else "cpu")
  
  if args.vis:
    from visdom import Visdom
    viz = Visdom(port=args.port)
    win = None
    
  # The online env in AItrans, it should have the observation space, action space and so on
  # We should step into the depth of envs.py in the github doc, and extract the format of observation
  # and action space 
  envs = 
  
  actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                        base_kwargs={'recurrent': args.recurrent_policy})
  actor_critic.to(device)
  
  # choose the algorithm, now we only have a2c
  if args.algo == 'a2c':
    agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                           args.entropy_coef, lr=args.lr,
                           eps=args.eps, alpha=args.alpha,
                           max_grad_norm=args.max_grad_norm)
    
  rollouts = RolloutStorage(args.num_steps, args.num_processes, 
                            envs.observation_space.shape, envs.action_space, 
                            actor_critic.recurrent_hidden_state_size) 
  
  # the initial observation
  obs = 
  rollouts.obs[0].copy_(obs)
  rollouts.to(device)
  
  episode_reward = deque(maxlen=10)
  start = time.time()
