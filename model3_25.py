#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torchvision
from skimage import io
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

torch.cuda.empty_cache()

momentum = 0.5

random_seed = 1
#torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

path = '/home/rwf8829/data/Gauguin_HSI_full.tif'

##GPU 
gpu_no = 0
use_cuda = True
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
data = torch.tensor(io.imread(path).astype(np.float32), device = device , dtype = torch.float32)
data.to(device)

data = data/torch.max(data)
data = data.unsqueeze(0)


data = F.interpolate(data, size=[384, 256])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(240, 214, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(214, 188, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(188, 162, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(162, 136, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(136, 110, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(110, 84, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(84, 58, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(58, 32, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.ConvTranspose2d(32, 58, kernel_size=2, stride=2)
        self.conv10 = nn.ConvTranspose2d(58, 84, kernel_size=2, stride=2)
        self.conv11 = nn.ConvTranspose2d(84, 110, kernel_size=2, stride=2)
        self.conv12 = nn.ConvTranspose2d(110, 136, kernel_size=1, stride=1)
        self.conv13 = nn.ConvTranspose2d(136, 162, kernel_size=1, stride=1)
        self.conv14 = nn.ConvTranspose2d(162, 188, kernel_size=1, stride=1)
        self.conv15 = nn.ConvTranspose2d(188, 214, kernel_size=1, stride=1)
        self.conv16 = nn.ConvTranspose2d(214, 240, kernel_size=1, stride=1)
                
        self.pool = nn.MaxPool2d(2, 2)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x
        
    def decode(self, x):
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.sigmoid(self.conv16(x))
        return x
    
    def forward(self, x):
        x = self.decode(self.encode(x))
        return x
        
                
Model = Net()
print(Model)
if torch.cuda.is_available():
    Model.cuda()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
criterion = nn.MSELoss()


en_output = Model.encode(data)
print(en_output.shape)
output = Model(data)
print(output.shape)
print(data.shape)

# training

loss_curve = []

for count in range(1200):
    optimizer.zero_grad()
    output = Model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {} \tLoss: {:f}'.format(count, loss.item()))
    loss_curve.append(loss.item())
    if count % 200 == 0:
        torch.save(Model.state_dict(), '/home/rwf8829/model/model3_25weights.pth')
        
torch.save(loss_curve, '/home/rwf8829/model/loss_curve14.pth')

