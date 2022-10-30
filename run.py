# Authors	: Cristopher McIntyre Garcia,
# Email	    : cmcin019@uottawa.ca
# S-N	    : 300025114

# Imports
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from tqdm import tqdm 
from os import system
import os

from datasets import get_mnist_dataloaders
from models import vit_tiny
from models.vision_transformer import vit_large

# Torch imports 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import torchvision


CUDA=torch.cuda.is_available()

# Set device
device = torch.device("cuda:0" if CUDA else "cpu")
print(f'Device: {device}')

# Load dataset
train_loader, test_loader = get_mnist_dataloaders()

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Simple configurations
model = vit_large(in_chans=1, num_classes=10)

def train(model):
	model.to(device=device)
	acc_list = []
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	accuracy = 0
	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.__class__.__name__}')
		print(f'Epoch {epoch}: {accuracy}')
		for _, (data, targets) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)
			targets = targets.to(device=device)

			scores = model(data)
			loss = criterion(scores, targets)

			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

		accuracy = model_accuracy(model)
		acc_list.append(accuracy)
	
	print(f'Final accuracy: {accuracy}')
	if device=='cuda:0':
		model.to(device='cpu')
	return acc_list

def model_accuracy(model):
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for images, labels in train_loader:
			images = images.to(device=device)
			labels = labels.to(device=device)

			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			if device=='cuda:0':
				correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
			else:
				correct += (predicted==labels).sum().item()
			
		TestAccuracy = 100 * correct / total

	model.train()
	return(TestAccuracy)



acc_list = train(model)
system('cls' if os.name == 'nt' else 'clear')
print(model.__class__.__name__)
for acc in range(len(acc_list)):
	if acc % 2 == 0:
		print(f'Epoch {acc+1}: \t{str(acc_list[acc])}')