# reference from paper: Deep learning-based human action recognition to leverage context awareness in collaborative assembly
import torch
from torch import nn
from einops import rearrange
from torchinfo import summary
class Bottleneck(nn.Module):
    """
    Bottleneck block structure used in ResNet 50. 
    As mentioned in Section 4. 2D ConvNet baseline (C2D), 
    all convolutions are in essence 2D kernels that prcoess the input frame-by-frame 
    (implemented as (1 x k x k) kernels). 
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=(0, 1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, 4*planes, kernel_size=(1, 3, 3), stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(4*planes)
        self.conv3 = nn.Conv3d(4*planes, inplanes, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        print(out.shape, identity.shape)
        
        out += identity
        out = self.relu(out)

        return out


class Res_RNN(nn.Module):
    def __init__(self, block, blocks_layer_num,inplane, plane, num_class, lstm_hiddensize):
        super().__init__()

        self.moulelist = nn.ModuleList([block(inplanes=inplane, planes=plane) for i in range(blocks_layer_num)]) # return [N, C, D, H, W]
        #The input size of lstm is [N, D, H] when batch_first is true
        self.lstm = nn.LSTM(input_size=3*112*112,hidden_size=lstm_hiddensize, batch_first=True, bidirectional=True, num_layers=2)
        #out.shape = [N, D ,2*hidden_size]

        self.linear1 = nn.Linear(in_features=2*lstm_hiddensize, out_features=lstm_hiddensize)
        self.linear2 = nn.Linear(in_features=lstm_hiddensize, out_features=num_class)
    
    def forward(self, x):
        for layer in self.moulelist:
            x = layer(x)
        x = rearrange(x, 'N C D H W -> N D (C H W)')
        x,_ = self.lstm(x)
        x = x[:,0,:]
        x =self.linear1(x)
        x = self.linear2(x)

        return x

if __name__ == '__main__':
    x = torch.randn(size=(2,3,16,112,112))
    net = Res_RNN(block=Bottleneck, blocks_layer_num=4,inplane=3,plane=512,num_class=12,lstm_hiddensize=512)
    summary(model=net,input_size=(2,3,16,112,112))