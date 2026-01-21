import torch
from torch.utils.data import Dataset
import os.path as osp

class PhaseIdentification(Dataset):
    def __init__(self, datadir, mode='train'):

        self.datadir = datadir
        self.mode = mode

        assert mode in ['train','test','val']
    
        fn_x = osp.join(datadir,'%s_signals.pt'%(mode))
        self.label = torch.load(osp.join(datadir,'%s_labels.pt'%mode))

        self.num_sample = self.label.shape[0]

        self.data = torch.load(fn_x)
        self.data = self.data.view(self.data.shape[0],1,-1)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
