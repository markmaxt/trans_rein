import argparse

import torch

def get_args():
  parser = argparse.ArgumentParser(description='RL')
  parser.add_argument('--port', type=int, default=8097, 
                      help='port to run the server on (default: 8097)') 
