import math as m
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from functions import *

def plot_run(initial_skill_params, p_matrix, p, std_dev, N,  color, label):
    skill_params = [param.clone() for param in initial_skill_params]

    for i in range(N - p):
        next_skill_params = generate_next_skill_params(skill_params, p_matrix, p, std_dev)
        skill_params.append(next_skill_params)

    time_steps = np.linspace(0, len(skill_params), len(skill_params))
    time_steps = np.arange(len(skill_params))
    first_skill_params = [skill_param[0].item() for skill_param in skill_params]
    plt.plot(time_steps, first_skill_params, color=color, label=label)

num_players = 1
p = 10
std_dev = 0.001
N = 1000
normalize = True
initial_skill_params, p_matrix = setup(num_players, p)


if normalize:
    p_matrix = torch.divide(p_matrix, torch.sum(p_matrix))

# p_matrix *= 0.5

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange']
for i in range(4):
    plot_run(initial_skill_params, p_matrix, p, std_dev, N, colors[i], f'Run {i+1}')

plt.xlabel("Timestep")
plt.ylabel("Skill (Player 0)")
plt.title("AR Skill Evolution (4 Runs with Same Initial Conditions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
