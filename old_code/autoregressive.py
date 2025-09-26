import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math as m
import time
import numpy as np
import torch.nn as nn
import random

from torch.nn.modules import linear
from functions import *
from tqdm import tqdm
import pickle

from torch.utils.data import DataLoader, TensorDataset

#parameters

num_players = 10
p = 10
std_dev = 0.001
num_observations = 1000
dataset_size = 2000
generate_dataset = False
batch_size = 200
epochs = 100

#setup

players = list(range(0,num_players))

linearly_indexed_matrix = np.zeros((num_players, num_players), dtype=np.float32)

count = 0
for i in range(num_players):
    for j in range(num_players):
        if i == j:
            continue

        linearly_indexed_matrix[i,j] = count
        count+=1


linearly_indexed_matrix = linearly_indexed_matrix / np.max(linearly_indexed_matrix)

#creating dataset
if generate_dataset:
    dataset = []
    p_matrices = []
    for _ in tqdm(range(dataset_size)):

        outcomes = []
        initial_skill_params, p_matrix = setup(num_players, p)
        skill_params = initial_skill_params
        p_matrices.append(p_matrix)

        for i in range(num_observations):
            next_skill_params = generate_next_skill_params(skill_params, p_matrix, p, std_dev)
            outcomes.append(play_game(next_skill_params, players, linearly_indexed_matrix))
            skill_params.append(next_skill_params)

        dataset.append(outcomes)

    dataset = torch.tensor(dataset)
    p_matrices = torch.stack(p_matrices,dim=0)
    labels = p_matrices
    torch.save(dataset, "dataset.pt")
    torch.save(labels, "labels.pt")
    observations = dataset

else:
    observations = torch.load("dataset.pt")
    labels = torch.load("labels.pt")


split = 1600
train_dataset = TensorDataset(observations[:split], labels[:split])
test_dataset = TensorDataset(observations[split:], labels[split:])

training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

mlp = MLP(num_players, num_observations, p).cuda()


optim = torch.optim.Adam(params=mlp.parameters(), lr=1e-3, weight_decay=0.001)

avg_losses = []
epochs_list = []
eval_losses = []
eval_epochs =[]

for epoch in range(epochs):

    if epoch % 5 == 0:
        #eval
        losses = []
        for batch in test_dataloader:

            observations, labels = batch

            observations = observations.cuda()
            labels = labels.cuda()

            predicted = mlp(observations)
            loss = F.mse_loss(predicted, labels)
            losses.append(loss.item())


        avg_loss = sum(losses) / len(losses)
        print("Test Loss: ", avg_loss)
        eval_losses.append(avg_loss)
        eval_epochs.append(epoch)

    loss_per_epoch = []
    for batch in training_dataloader:
        optim.zero_grad()

        observations, labels = batch

        observations = observations.cuda()
        labels = labels.cuda()

        predicted = mlp(observations)

        loss = F.mse_loss(predicted, labels)

        loss.backward()
        optim.step()
        loss_per_epoch.append(loss.item())

    if epoch % 5 == 0:
        avg_loss_per_epoch = sum(loss_per_epoch) / len(loss_per_epoch)
        avg_losses.append(avg_loss_per_epoch)
        epochs_list.append(epoch)


    if epoch % 5 == 0:
        print("Epoch: ", epoch, "loss: ", loss.item())

print(torch.div((predicted[0] - labels[0]) , predicted[0]))
# print(labels[0])

plt.figure()
plt.plot(epochs_list,avg_losses, label="train")
plt.plot(eval_epochs,eval_losses, label="eval")
plt.legend()
plt.title("Train losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("plot.jpg")
