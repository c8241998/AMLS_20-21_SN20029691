import torch
import torch.nn as nn



class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            SELayer(4),
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            SELayer(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            SELayer(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.model2 = nn.Sequential(
            nn.Linear(32*32, 32),
            nn.Linear(32, 5),
        )
    def forward(self, x):
        # x   b,4,256,256
        x=self.model(x)
        x=x.view(x.shape[0],-1)
        x=self.model2(x)
        return x

if __name__=='__main__':
    x = torch.ones(64,3,224,224).cuda()
    model = MyModule().cuda()
    y = model(x)
    print(y.shape)