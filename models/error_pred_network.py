import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt

class errorNet(nn.Module):
    def __init__(self, Nin, Nout):
        super(errorNet, self).__init__()

        #self.L1 = nn.Linear(Nin, 4)
        #self.L2 = nn.Linear(4, Nout)
        #self.L3= nn.Linear(256, Nout)

        self.L1 = nn.Linear(Nin, 256)
        self.L2 = nn.Linear(256, 512)
        self.L3 = nn.Linear(512, 256)
        self.L4 = nn.Linear(256, Nout)
        #self.L5 = nn.Linear(4, Nout)

    def forward(self, x):

        #print(x.shape)

        #x1 = torch.sigmoid(self.L1(x))
        x1 = torch.relu(self.L1(x))
        x2 = torch.relu(self.L2(x1))
        x3 = torch.relu(self.L3(x2))
        x4 = self.L4(x3)

        #x2 = self.L2(x1)
        #print(x3)

        return x4

class errorNet2(nn.Module):
    def __init__(self, Nin, Nout):
        super(errorNet2, self).__init__()

        #self.L1 = nn.Linear(Nin, 4)
        #self.L2 = nn.Linear(4, Nout)
        #self.L3= nn.Linear(256, Nout)

        self.L1 = nn.Linear(Nin, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.L2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.L3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.L4 = nn.Linear(256, Nout)
        #self.L5 = nn.Linear(4, Nout)

    def forward(self, x):

        #print(x.shape)

        #x1 = torch.sigmoid(self.L1(x))
        x1 = torch.relu(self.bn1(self.L1(x)))
        x2 = torch.relu(self.bn2(self.L2(x1)))
        x3 = torch.relu(self.bn3(self.L3(x2)))
        x4 = self.L4(x3)

        #x2 = self.L2(x1)
        #print(x3)

        return x4



"""
class errorNet(nn.Module):
    def __init__(self, Nin, Nout):
        super(errorNet, self).__init__()
        module_list = nn.ModuleList()
        k = 3

        for i in range(30):
            modules = nn.Sequential()

            if i == 0:
               modules.add_module('Conv2d', nn.Conv2d(in_channels=1,
                                                       out_channels=256,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=k // 2))

               modules.add_module('BatchNorm2d', nn.BatchNorm2d(256, momentum=0.03, eps=1E-4))
               modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

            else:
                if (i%2) == 0:
                    inc = 128
                    outc = 256
                else:
                    inc = 256
                    outc = 128

                modules.add_module('Conv2d', nn.Conv2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=k // 2))

                modules.add_module('BatchNorm2d', nn.BatchNorm2d(outc, momentum=0.03, eps=1E-4))
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))



            module_list.append(modules)

        linmodule = nn.Sequential()
        linmodule.add_module('Linear', nn.Linear(45056, Nout))
        module_list.append(linmodule)

        self.module_list = module_list

    def forward(self, x):
        #print("inforward")
        for i, module in enumerate(self.module_list):
            if i == 30:
                x = torch.reshape(x,(-1, 45056) )
            #print(i, x.shape)
            x = module(x)

        return x
"""
