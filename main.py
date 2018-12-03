import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arguments import get_args

# import algo
from arguments import get_args
from visualize import visdom_plot

args = get_args()

torch.manual_seed(args.seed)


def main():
  torch.set_num_threads(1)
  device = torch.device("cuda:0" if args.cuda else "cpu")
  
  if args.vis:
    from visdom import Visdom
    viz = Visdom(port=args.port)
