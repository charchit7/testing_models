'''
train swin transformer for cifar10
'''
################################################################# IMPORTS ###################################

from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import time
import math
import os
##############################################################################################################


###############################################   CONFIGS ######################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 
start_epoch = 0
NUM_CLASSES = 10
IMAGE_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1

###############################################################################################################

###############################################   DATA ######################################################
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# this will create a validation set with 50 examples from each class.

train_and_valid = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)


torch.manual_seed(43)
indices = list(range(len(train_and_valid))) # indices of the dataset
train_indices,val_indices = train_test_split(indices, test_size=0.1, stratify=train_and_valid.targets, random_state=101) 

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)


train_sampler.transform = transform_train
val_sampler.transform = transform_test


trainloader = torch.utils.data.DataLoader(
    train_and_valid, batch_size=256, shuffle=False, sampler=train_sampler)

testloader = torch.utils.data.DataLoader(
    train_and_valid, batch_size=100, shuffle=False, sampler=val_sampler)

#################################################################################################################

from resnet import *


net = ResNet50()

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    NUM_EPOCHS = 200
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/r50_ckpt.pth')
        best_acc = acc
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if epoch == math.floor((10/100)*NUM_EPOCHS) or epoch == math.floor((30/100)*NUM_EPOCHS) or epoch == math.floor((50/100)*NUM_EPOCHS) or epoch == math.floor((70/100)*NUM_EPOCHS):
        print('saving model at:', epoch)
        save_path = './checkpoint/epoch_'+str(epoch)+'_model.pth'
        torch.save(net.state_dict(),
                    save_path)


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()



