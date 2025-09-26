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
std_dev = 0.001
N = 20

initial_skill_params, p_matrix = setup(num_players, p)
skill_params = initial_skill_params

for i in range(N - p):
    next_skill_params = generate_next_skill_params(skill_params, p_matrix, p, std_dev)
    skill_params.append(next_skill_params)

# now, using perfect estimate of skill params, need to estimate parameters of autoregressive process
# X: previous p observations of autoregressive process
# y: next observation

X = torch.zeros((N - p, p))

for i in range(N - p):
    X[i, :] = torch.tensor(skill_params[i : i + p])  # sliding window


y = torch.tensor(skill_params[p:])
p_matrix = torch.squeeze(p_matrix)


phi = torch.pinverse(X.T @ X) @ (X.T @ y)

print(phi)
print(p_matrix)
