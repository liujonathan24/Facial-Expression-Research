import os
import torch
from torch import nn
import numpy as np
import
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#############################################
# Get Device for Training
# -----------------------
# We want to be able to train our model on a hardware accelerator like the GPU,
# if it is available. Let's check to see if
# `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_ is available, else we
# continue to use the CPU.

def modell(Classifier):
    def __init__(self, labels=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(3 * 64 * 64, labels)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc(out)
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")nepochs = 5

losses = np.zeros(nepochs)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(modell.parameters(), lr = 0.001)

for epoch in range(nepochs):

    running_loss = 0.0
    n = 0

    for data in train_loader:

        #single batch
        if(n == 1):
            break;

        inputs, labels = data

        optimizer.zero_grad()

        outputs = modell(inputs)

        #loss = loss_fn(outputs, labels)
        loss = loss_fn(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n += 1

    losses[epoch] = running_loss / n
    print(f"epoch: {epoch+1} loss: {losses[epoch] : .3f}")