from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import pandas as pd
from skimage import io
import torch
import pickle

class CartoonDataset(Dataset):
    def __init__(self,dir,mode='train',split=8000):
        super().__init__()
        self.mode = mode
        self.dir = dir
        self.csv = pd.read_csv(dir+'labels.csv')
        self.split = split
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize([256,256]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5, 0.5])
                                                                  ])
        if mode=='train':
            self.len = 8000
        elif mode=='val':
            self.len = 2000
        else:
            self.len = 2500

    def __getitem__(self,index):

        index = index if self.mode != 'val' else self.split + index
        image = io.imread(self.dir+'img/'+str(index)+'.png')
        image = self.transform(image)
        info = self.csv.loc[index].values[0].split('\t')
        eye, face = info[1], info[2]
        label = torch.tensor([int(face)],dtype=torch.long)

        sample = {
            'image':image,
            'label': label,
            'info':info
        }
        return sample

    def __len__(self):
        return self.len

if __name__ == '__main__':
    dataset = CartoonDataset(mode='val')
    for i in range(len(dataset)):
        sample = dataset[i]
        img = sample['image']
        label = sample['label']
        info = sample['info']
        print(img.shape)
        print(label.shape)
        print(info)
        # print(img)
        break