import torch
from torch import nn
import math
import numpy as np

class CNN_NFhead(nn.Module):
    def __init__(self, n_layer,hidden_dim=64,ks=15,dim=4501):
        super(CNN_NFhead, self).__init__()
        # assert n_layer<=6
        self.features = []
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        for i in range(n_layer):
            if i==0:
                self.features.append(nn.Sequential(nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=ks, stride=1, padding=ks//2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1)))
                dim = math.ceil(dim/2)
            else:
                self.features.append(nn.Sequential(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=ks, stride=1, padding=ks//2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1)))
                dim = math.ceil(dim/2)
        self.dim = dim
        self.features = nn.Sequential(*self.features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        return x

class CNN_NF(nn.Module):
    def __init__(self, n_classes, n_layer,hidden_dim=64,ks=15):
        super(CNN_NF, self).__init__()
        # assert n_layer<=6
        self.convs = nn.ModuleList()
        self.n_layer = n_layer
        self.features = CNN_NFhead(n_layer=n_layer,hidden_dim=hidden_dim,
                                   ks=ks)
                                
        self.flatten = nn.Sequential(nn.Flatten())
        self.fc = nn.Linear(in_features=self.features.dim*hidden_dim, out_features=n_classes)

    def forward(self, x):
        x= x[0]
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNN_NF_mlp(nn.Module):
    def __init__(self, n_classes, n_layer,hidden_dim=64,ks=15,mlp_drop=0):
        super(CNN_NF_mlp, self).__init__()
        # assert n_layer<=6
        self.convs = nn.ModuleList()
        self.n_layer = n_layer
        self.features = CNN_NFhead(n_layer=n_layer,hidden_dim=hidden_dim,
                                   ks=ks)
                                
        self.flatten = nn.Sequential(nn.Flatten())
        self.head = nn.Sequential(nn.Flatten(),nn.Dropout(p=mlp_drop),
                                 nn.Linear(self.features.dim*hidden_dim,2500),nn.ReLU(),nn.Dropout(p=mlp_drop),
                                 nn.Linear(2500,1000),nn.ReLU(),nn.Dropout(p=mlp_drop),
                                 nn.Linear(1000,n_classes))

    def forward(self, x):
        x= x[0]
        x = self.features(x)
        x = self.flatten(x)
        x = self.head(x)
        return x