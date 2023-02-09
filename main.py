import os

import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import cv2
from model import PyramidGroupTransformer as PGT
import copy
from tqdm import tqdm
import time
from torchinfo import summary


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self, im_size, num_classes):
        super().__init__()
        self.pgt = PGT(im_size)
        self.clf = nn.Sequential(
            nn.Linear(im_size ** 2 // 32 ** 2 * 512, 2000),
            nn.ReLU(),
            nn.Linear(2000, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        # self.clf = nn.Linear(im_size ** 2 // 32 ** 2 * 512, num_classes)

    def forward(self, x):
        out1, out2, out3, out4 = self.pgt(x)
        logits = self.clf(out4.reshape(x.shape[0], -1))
        return logits


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    val_acc_history = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.permute(0, 2, 3, 1))
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} loss: {epoch_loss} acc: {epoch_acc}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'working/fashion_{best_acc:.4f}.txt')
            if phase == 'test':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s ')
    print(f'Best eval acc: {best_acc:.4f}')

    model.load_state_dict(best_model_weights)

    return model, val_acc_history

def main():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    data = torchvision.datasets.FashionMNIST(r'E:\Desktop\segmentation\fashion-mnist', transform=transform, download=True)
    train, test = torch.utils.data.random_split(data, (int(len(data) * 0.8), int(len(data) * 0.2)))
    train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, drop_last=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=True, drop_last=False, num_workers=2)

    model = Model(32, 10)
    summary(model)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.0005)

    dataloaders = {'train': train_loader, 'test': test_loader}
    model, hist = train_model(model, dataloaders, criterion, optim, num_epochs=20)


if __name__ == '__main__':
    main()
