from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm.auto import tqdm
class MySkeletonDataset(Dataset):
    def __init__(self, data_root_path:str):

        super(MySkeletonDataset).__init__()
        self.data_root_path = data_root_path
        self.Skeleton, self.labels = self.data_load(data_path=self.data_root_path)
        self.len = len(self.labels)
        print('数据已经准备好了....')
    def __getitem__(self, index) :

        return self.Skeleton[index], self.labels[index]
    def __len__(self):
        return self.len

    def data_load(self, data_path):
        path_list = os.listdir(data_path)[0:4]
        X_Skeleton = []
        y = []
        for path in tqdm(path_list):
            Skeleton_data = torch.from_numpy(np.load(f'{data_path}/{path}/{path}_skeletons.npy'))
            X_Skeleton.append(Skeleton_data)
            label_data = torch.from_numpy(np.load(f'{data_path}/{path}/{path}_label.npy'))
            y.append(label_data)
        return  torch.cat(X_Skeleton,dim=0), torch.cat(y,dim=0)