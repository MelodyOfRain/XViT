import torch
from torch.utils.data import Dataset
import os.path as osp

class PhaseIdentification(Dataset):
    def __init__(self, datadir, datasize,mode='train',test_frac=0.1,suffix='signals'):

        self.datadir = datadir
        self.mode = mode

        assert mode in ['train','test','val']
    
        fn_x = osp.join(datadir,'%s_%s.pt'%(mode,suffix))
        self.label = torch.load(osp.join(datadir,'%s_labels.pt'%mode))

        self.num_sample = int(self.label.shape[0]*datasize)

        self.data = torch.load(fn_x)[:self.num_sample,]
        self.data = self.data.view(self.data.shape[0],1,-1)
        self.label = self.label[:self.num_sample,]
        self.label = torch.sum(self.label, dim=-1)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
   
    def __len__(self):
        return len(self.data)
