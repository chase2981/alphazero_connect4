import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from IPython.display import HTML
from base64 import b64encode

config_dict = {
    'device': torch.device('cuda') if torch.cuda.is_available() else 'cpu',
    'n_filters': 128,
    'n_res_blocks': 8,
    'exploration_constant': 2.5,  # Slightly more exploration
    'temperature': 1.25,
    'dirichlet_alpha': 0.8,      # Less noise
    'dirichlet_eps': 0.2,        # Lower noise contribution
    'learning_rate': 0.001,
    'training_epochs': 80,
    'games_per_epoch': 100,
    'minibatch_size': 128,
    'n_minibatches': 4,
    'mcts_start_search_iter': 50, # Increased initial search iterations
    'mcts_max_search_iter': 300,  # Deeper max search
    'mcts_search_increment': 5,   # Gradual increases in search depth
}


# Convert to a struct esque object
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

config = Config(config_dict)