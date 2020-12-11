from B1.dataset import CartoonDataset
from B1.models import *
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn,optim
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def main(root_dir):
    testset = CartoonDataset(mode='test',dir=root_dir+'../Datasets/cartoon_set_test/')
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=2,pin_memory=True)
    model = MyModule()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load("B1/model.pth"))

    cnt_test = 0.
    for sample in testloader:
        image, label = sample['image'].cuda(), sample['label'].cuda()
        model.eval()
        pred = model(image)
        batch = image.shape[0]
        for i in range(batch):
            pred_i = torch.argmax(pred[i])
            label_i = label[i]
            if (abs(pred_i - label_i) < 0.01):
                cnt_test = cnt_test + 1.

    test_acc = cnt_test / 2500.
    return test_acc