from B1.dataset import CartoonDataset
from B1.models import *
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def train(model,criterion,optimizer,trainloader,valloader,schedular,epochs,config):
    best_acc_train,best_acc_val = 0,0
    loss_train,loss_val,acc_train,acc_val = [],[],[],[]

    for epoch in range(epochs):
        time_start = time.time()
        losses = np.array([])
        for i,sample in enumerate(trainloader):
            image,label = sample['image'].cuda(),sample['label'].cuda()
            model.train()
            pred = model(image)
            loss = criterion(pred, label.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses = np.append(losses,loss.item())

        losses_val = np.array([])
        for sample in valloader:
            image, label = sample['image'].cuda(), sample['label'].cuda()
            model.eval()
            pred = model(image)
            loss = criterion(pred, label.squeeze(-1))
            losses_val = np.append(losses_val,loss.item())


        print(epoch,':   training loss —— ', losses.mean(),
              '       validation loss —— ', losses_val.mean(),
              '       lr ——',optimizer.state_dict()['param_groups'][0]['lr'])

        train_acc,val_acc = inference(model=model,trainloader=trainloader,valloader=valloader)
        if val_acc>best_acc_val:
            torch.save(model.state_dict(), 'B1/model.pth')
            best_acc_train = train_acc
            best_acc_val = val_acc

        loss_train.append(losses.mean())
        loss_val.append(losses_val.mean())
        acc_train.append(train_acc)
        acc_val.append(val_acc)

        if config['if-schedular'] == "true":
            schedular.step()

        time_end = time.time()
        print("time cost: ", time_end - time_start)

    plt.plot(loss_train, 'r', label='train')
    plt.plot(loss_val, 'b', label='val')
    plt.legend(loc='upper right')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(range(0, epochs, 2))
    plt.savefig('B1/loss.png')
    plt.cla()
    plt.plot(acc_train, 'r', label='train')
    plt.plot(acc_val, 'b', label='val')
    plt.legend(loc='upper right')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.xticks(range(0, epochs, 2))
    plt.savefig('B1/acc.png')

    return best_acc_train,best_acc_val

def read_config(dir):
    with open(dir, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict

def main(root_dir):
    config = read_config(root_dir+'./config.json')
    print('preparing dataset')
    trainset = CartoonDataset(mode='train',dir=root_dir+'../Datasets/cartoon_set/')
    valset = CartoonDataset(mode='val',dir=root_dir+'../Datasets/cartoon_set/')
    trainloader = DataLoader(trainset, batch_size=config['batch-size'], shuffle=True, num_workers=config['num-workers'],pin_memory=True)
    valloader = DataLoader(valset, batch_size=config['batch-size'], shuffle=True, num_workers=config['num-workers'],pin_memory=True)
    print('prepared successfully')
    model = MyModule()
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config['learning-rate'],weight_decay=config['weight-decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config['schedular-step-size'], gamma=config['schedular-gamma'])


    best_acc_train,best_acc_val = train(model=model,criterion=criterion,optimizer=optimizer,epochs=config['epochs'],
                                        trainloader=trainloader,valloader=valloader,schedular=scheduler,config=config)

    return best_acc_train,best_acc_val

def inference(model,trainloader,valloader):
    cnt_train = 0.
    for sample in trainloader:
        image, label = sample['image'].cuda(), sample['label'].cuda()
        model.eval()
        pred = model(image)
        batch = image.shape[0]
        for i in range(batch):
            pred_i = torch.argmax(pred[i])
            label_i = label[i]
            if (abs(pred_i - label_i) < 0.01):
                cnt_train = cnt_train + 1.
    cnt_val=0.
    for sample in valloader:
        image, label = sample['image'].cuda(), sample['label'].cuda()
        model.eval()
        pred = model(image)
        batch = image.shape[0]
        for i in range(batch):
            pred_i = torch.argmax(pred[i])
            label_i = label[i]
            if(abs(pred_i-label_i)<0.01):
                cnt_val=cnt_val+1.
    train_acc, val_acc = cnt_train/8000.,cnt_val/2000.
    print('train-acc: ',train_acc,'      val-acc: ',val_acc)
    return train_acc,val_acc

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()