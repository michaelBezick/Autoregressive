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

num_players = 1
p = 10
std_dev = 0.1
N = 1000

initial_skill_params, p_matrix = setup(num_players, p)
skill_params = initial_skill_params

for i in range(N - p):
    next_skill_params = generate_next_skill_params(skill_params, p_matrix, p, std_dev)
    skill_params.append(next_skill_params)

plt.figure()
time_steps = np.linspace(0, len(skill_params), len(skill_params))
first_skill_params = [skill_param[0] for skill_param in skill_params]
plt.plot(time_steps, first_skill_params)
plt.show()
