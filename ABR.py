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

import algo
from arguments import get_args
from model import Policy
from storage import RolloutStorage
from utils import update_linear_schedule, get_vec_normalize

# import algo
from arguments import get_args
from visualize import visdom_plot

NN_MODEL = None

class Algorithm:
    def __init__(self):
        self.buffer_size = 0

    # Initial
    def Initial(self):
        # Initial my session or something
        args = get_args()

        assert args.algo in ['a2c', 'ppo']
        if args.recurrent_policy:
            assert args.algo in ['a2c', 'ppo'], \
                'Recurrent policy is not implemented for ACKTR'

        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        try:
            os.makedirs(args.log_dir)
        except OSError:
            files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

        eval_log_dir = args.log_dir + "_eval"

        try:
            os.makedirs(eval_log_dir)
        except OSError:
            files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

        # Define my algorithm
        def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
                S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, end_of_video, cdn_newest_id, download_id,
                cdn_has_frame, IntialVars):
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
            elif args.algo == 'ppo':
                agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                                 args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                 eps=args.eps,
                                 max_grad_norm=args.max_grad_norm)
            elif args.algo == 'acktr':
                agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                                       args.entropy_coef, acktr=True)

            rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                      envs.observation_space.shape, envs.action_space,
                                      actor_critic.recurrent_hidden_state_size)

            # the initial observation
            obs =
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)

            episode_reward = deque(maxlen=10)
            start = time.time()
            for j in range(num_updates):

                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    if args.algo == "acktr":
                        # use optimizer's learning rate since it's hard-coded in kfac.py
                        update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
                    else:
                        update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

                if args.algo == 'ppo' and args.use_linear_lr_decay:
                    agent.clip_param = args.clip_param * (1 - j / float(num_updates))

                for step in range(args.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step],
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])


