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


class SoftMax_Regression(nn.Module):
	def __init__(self, in_channels=1, out_channels=10, drop=False, bn=False):
		super(SoftMax_Regression, self).__init__()
		self.name = f'SoftMax_Regression - dropout={drop} bn={bn}'
		self.linear = nn.Linear(28*28, 10) 
		self.soft_max = nn.Softmax(1)
		self.drop = drop
		self.dropout = nn.Dropout(p=0.2)
		self.bn = bn
		self.batchnorm = nn.BatchNorm1d(28*28)

	def forward(self, image):
		a = image.view(-1, 28*28)
		if self.drop:
			a = self.dropout(a)
		if self.bn:
			a = self.batchnorm(a)
		a = self.linear(a)
		a = self.soft_max(a)
		return a
		
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.name = 'CNN'
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 120)
		self.fc3 = nn.Linear(120, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Simple configurations
model = SoftMax_Regression()

def train(model):
	model.to(device=device)
	acc_list = []
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	accuracy = 0
	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.name}')
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
print(model.name)
for acc in range(len(acc_list)):
	if acc % 2 == 0:
		print(f'Epoch {acc+1}: \t{str(acc_list[acc])}')