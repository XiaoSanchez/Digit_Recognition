import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F 
from tqdm.notebook import tqdm
import utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchvision import datasets 
from torchvision import transforms as T
from PIL import ImageOps
import random
class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
    def __call__(self, img):
        direction = random.randint(0, 3)  
        if direction == 0:  
            img = ImageOps.expand(img, border=(self.shift, 0, 0, 0))  
        elif direction == 1:  
            img = ImageOps.expand(img, border=(0, 0, self.shift, 0))  
        elif direction == 2:  
            img = ImageOps.expand(img, border=(0, self.shift, 0, 0))  
        elif direction == 3:  
            img = ImageOps.expand(img, border=(0, 0, 0, self.shift))  
        return img
shift = RandomShift(shift=2)  
train_augs = T.Compose([
    shift,
    T.RandomCrop(28),  
    T.ToTensor(),
    T.Normalize(mean = 0.5, std = 0.5)
])
valid_augs = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = 0.5, std = 0.5)
])
trainset = datasets.MNIST('./', download = False, train = True, transform = train_augs)
testset = datasets.MNIST('./', download = False, train = False, transform = valid_augs)
trainset, validset = torch.utils.data.random_split(trainset, [50000, 10000])
idx = 1
image, label = trainset[idx]
plt.imshow(image.permute(1, 2, 0), cmap = 'gray')
plt.title(label)
from torch.utils.data import DataLoader
bs = 64
trainloader = DataLoader(trainset, batch_size = bs, shuffle = True)
validloader = DataLoader(validset, batch_size = bs)
testloader = DataLoader(testset, batch_size = bs)
for images, labels in trainloader:
    break
from models import DigitModel
model = DigitModel()
model.to(device)
def train_fn(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += utils.multiclass_accuracy(logits, labels)
    return total_loss / len(dataloader), total_acc / len(dataloader)
def eval_fn(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_acc += utils.multiclass_accuracy(logits, labels)
        return total_loss / len(dataloader), total_acc / len(dataloader)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.003)
best_valid_loss = np.Inf
for i in range(10):
    train_loss, train_acc = train_fn(model, trainloader, criterion, optimizer)
    valid_loss, valid_acc = eval_fn(model, trainloader, criterion)
    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'best_weights.pt')
        best_valid_loss = valid_loss
image, label = testset[0]
weights = torch.load('best_weights.pt')
model.load_state_dict(weights)
model.eval()
with torch.no_grad():
    logits = model(image.unsqueeze(0)) 
    ps = torch.nn.Softmax(dim = 1)(logits)[0]
    utils.view_classify(image, ps)
