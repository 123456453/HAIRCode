import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
#borrow from  https://github.com/tea1528/Non-Local-NN-Pytorch

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

#shape of Non_local block loaded by summary
# ==========================================================================================
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# NLBlockND                                [2, 3, 4, 20, 20]         --
# ├─Conv3d: 1-1                            [2, 1, 4, 20, 20]         4
# ├─Conv3d: 1-2                            [2, 1, 4, 20, 20]         4
# ├─Conv3d: 1-3                            [2, 1, 4, 20, 20]         4
# ├─Sequential: 1-4                        [2, 1, 1600, 1600]        --
# │    └─Conv2d: 2-1                       [2, 1, 1600, 1600]        3
# │    └─ReLU: 2-2                         [2, 1, 1600, 1600]        --
# ├─Conv3d: 1-5                            [2, 3, 4, 20, 20]         6
# ==========================================================================================
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
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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
        
        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """C2D with ResNet 50 backbone.
    The only operation involving the temporal domain are the pooling layer after the second residual block.
    For more details of the structure, refer to Table 1 from the paper. 
    Padding was added accordingly to match the correct dimensionality.
    """
    def __init__(self, block, layers, num_classes=12, non_local=False):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        
        # first convolution operation has essentially 2D kernels
        # output: 64 x 16 x 112 x 112
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=2, padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # output: 64 x 8 x 56 x 56
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        
        # output: 256 x 8 x 56 x 56
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, d_padding=0)
        
        # pooling on temporal domain
        # output: 256 x 4 x 56 x 56
        self.pool_t = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1))
        
        # output: 512 x 4 x 28 x 28
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=(2, 1, 1))
        
        # add one non-local block here
        # output: 1024 x 4 x 14 x 14
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=(2, 1, 1), non_local=non_local)

        # output: 2048 x 4 x 7 x 7
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=(2, 1, 1))
        
        # output: 2048 x 1
        self.avgpool = nn.AvgPool3d(kernel_size=(4, 7, 7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, padding=(0, 1, 1), d_padding=(2, 0, 0), non_local=False):
        downsample = nn.Sequential(
                            nn.Conv3d(self.inplanes, planes * block.expansion, 
                                      kernel_size=1, stride=stride, padding=d_padding, bias=False), 
                            nn.BatchNorm3d(planes * block.expansion)
                        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, padding, downsample))
        self.inplanes = planes * block.expansion
        
        last_idx = blocks
        if non_local:
            last_idx = blocks - 1
            
        for i in range(1, last_idx):
            layers.append(block(self.inplanes, planes))
        
        # add non-local block here
        if non_local:
            layers.append(NLBlockND(in_channels=1024, dimension=3))
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.pool_t(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet3D50(non_local=False, **kwargs):
    """Constructs a C2D ResNet-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], non_local=non_local, **kwargs)
    return model

#shape of Resnet3D50 with non-local
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ResNet3D                                 [1, 12]                   --
# ├─Conv3d: 1-1                            [1, 64, 16, 112, 112]     9,408
# ├─BatchNorm3d: 1-2                       [1, 64, 16, 112, 112]     128
# ├─ReLU: 1-3                              [1, 64, 16, 112, 112]     --
# ├─MaxPool3d: 1-4                         [1, 64, 7, 55, 55]        --
# ├─Sequential: 1-5                        [1, 256, 7, 55, 55]       --
# │    └─Bottleneck: 2-1                   [1, 256, 7, 55, 55]       --
# │    │    └─Conv3d: 3-1                  [1, 64, 7, 55, 55]        4,096
# │    │    └─BatchNorm3d: 3-2             [1, 64, 7, 55, 55]        128
# │    │    └─ReLU: 3-3                    [1, 64, 7, 55, 55]        --
# │    │    └─Conv3d: 3-4                  [1, 64, 7, 55, 55]        36,864
# │    │    └─BatchNorm3d: 3-5             [1, 64, 7, 55, 55]        128
# │    │    └─ReLU: 3-6                    [1, 64, 7, 55, 55]        --
# │    │    └─Conv3d: 3-7                  [1, 256, 7, 55, 55]       16,384
# │    │    └─BatchNorm3d: 3-8             [1, 256, 7, 55, 55]       512
# │    │    └─Sequential: 3-9              [1, 256, 7, 55, 55]       16,896
# │    │    └─ReLU: 3-10                   [1, 256, 7, 55, 55]       --
# │    └─Bottleneck: 2-2                   [1, 256, 7, 55, 55]       --
# │    │    └─Conv3d: 3-11                 [1, 64, 7, 55, 55]        16,384
# │    │    └─BatchNorm3d: 3-12            [1, 64, 7, 55, 55]        128
# │    │    └─ReLU: 3-13                   [1, 64, 7, 55, 55]        --
# │    │    └─Conv3d: 3-14                 [1, 64, 7, 55, 55]        36,864
# │    │    └─BatchNorm3d: 3-15            [1, 64, 7, 55, 55]        128
# │    │    └─ReLU: 3-16                   [1, 64, 7, 55, 55]        --
# │    │    └─Conv3d: 3-17                 [1, 256, 7, 55, 55]       16,384
# │    │    └─BatchNorm3d: 3-18            [1, 256, 7, 55, 55]       512
# │    │    └─ReLU: 3-19                   [1, 256, 7, 55, 55]       --
# │    └─Bottleneck: 2-3                   [1, 256, 7, 55, 55]       --
# │    │    └─Conv3d: 3-20                 [1, 64, 7, 55, 55]        16,384
# │    │    └─BatchNorm3d: 3-21            [1, 64, 7, 55, 55]        128
# │    │    └─ReLU: 3-22                   [1, 64, 7, 55, 55]        --
# │    │    └─Conv3d: 3-23                 [1, 64, 7, 55, 55]        36,864
# │    │    └─BatchNorm3d: 3-24            [1, 64, 7, 55, 55]        128
# │    │    └─ReLU: 3-25                   [1, 64, 7, 55, 55]        --
# │    │    └─Conv3d: 3-26                 [1, 256, 7, 55, 55]       16,384
# │    │    └─BatchNorm3d: 3-27            [1, 256, 7, 55, 55]       512
# │    │    └─ReLU: 3-28                   [1, 256, 7, 55, 55]       --
# ├─MaxPool3d: 1-6                         [1, 256, 3, 55, 55]       --
# ├─Sequential: 1-7                        [1, 512, 4, 28, 28]       --
# │    └─Bottleneck: 2-4                   [1, 512, 4, 28, 28]       --
# │    │    └─Conv3d: 3-29                 [1, 128, 3, 55, 55]       32,768
# │    │    └─BatchNorm3d: 3-30            [1, 128, 3, 55, 55]       256
# │    │    └─ReLU: 3-31                   [1, 128, 3, 55, 55]       --
# │    │    └─Conv3d: 3-32                 [1, 128, 4, 28, 28]       147,456
# │    │    └─BatchNorm3d: 3-33            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-34                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-35                 [1, 512, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-36            [1, 512, 4, 28, 28]       1,024
# │    │    └─Sequential: 3-37             [1, 512, 4, 28, 28]       132,096
# │    │    └─ReLU: 3-38                   [1, 512, 4, 28, 28]       --
# │    └─Bottleneck: 2-5                   [1, 512, 4, 28, 28]       --
# │    │    └─Conv3d: 3-39                 [1, 128, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-40            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-41                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-42                 [1, 128, 4, 28, 28]       147,456
# │    │    └─BatchNorm3d: 3-43            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-44                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-45                 [1, 512, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-46            [1, 512, 4, 28, 28]       1,024
# │    │    └─ReLU: 3-47                   [1, 512, 4, 28, 28]       --
# │    └─Bottleneck: 2-6                   [1, 512, 4, 28, 28]       --
# │    │    └─Conv3d: 3-48                 [1, 128, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-49            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-50                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-51                 [1, 128, 4, 28, 28]       147,456
# │    │    └─BatchNorm3d: 3-52            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-53                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-54                 [1, 512, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-55            [1, 512, 4, 28, 28]       1,024
# │    │    └─ReLU: 3-56                   [1, 512, 4, 28, 28]       --
# │    └─Bottleneck: 2-7                   [1, 512, 4, 28, 28]       --
# │    │    └─Conv3d: 3-57                 [1, 128, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-58            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-59                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-60                 [1, 128, 4, 28, 28]       147,456
# │    │    └─BatchNorm3d: 3-61            [1, 128, 4, 28, 28]       256
# │    │    └─ReLU: 3-62                   [1, 128, 4, 28, 28]       --
# │    │    └─Conv3d: 3-63                 [1, 512, 4, 28, 28]       65,536
# │    │    └─BatchNorm3d: 3-64            [1, 512, 4, 28, 28]       1,024
# │    │    └─ReLU: 3-65                   [1, 512, 4, 28, 28]       --
# ├─Sequential: 1-8                        [1, 1024, 4, 14, 14]      --
# │    └─Bottleneck: 2-8                   [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-66                 [1, 256, 4, 28, 28]       131,072
# │    │    └─BatchNorm3d: 3-67            [1, 256, 4, 28, 28]       512
# │    │    └─ReLU: 3-68                   [1, 256, 4, 28, 28]       --
# │    │    └─Conv3d: 3-69                 [1, 256, 4, 14, 14]       589,824
# │    │    └─BatchNorm3d: 3-70            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-71                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-72                 [1, 1024, 4, 14, 14]      262,144
# │    │    └─BatchNorm3d: 3-73            [1, 1024, 4, 14, 14]      2,048
# │    │    └─Sequential: 3-74             [1, 1024, 4, 14, 14]      526,336
# │    │    └─ReLU: 3-75                   [1, 1024, 4, 14, 14]      --
# │    └─Bottleneck: 2-9                   [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-76                 [1, 256, 4, 14, 14]       262,144
# │    │    └─BatchNorm3d: 3-77            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-78                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-79                 [1, 256, 4, 14, 14]       589,824
# │    │    └─BatchNorm3d: 3-80            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-81                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-82                 [1, 1024, 4, 14, 14]      262,144
# │    │    └─BatchNorm3d: 3-83            [1, 1024, 4, 14, 14]      2,048
# │    │    └─ReLU: 3-84                   [1, 1024, 4, 14, 14]      --
# │    └─Bottleneck: 2-10                  [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-85                 [1, 256, 4, 14, 14]       262,144
# │    │    └─BatchNorm3d: 3-86            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-87                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-88                 [1, 256, 4, 14, 14]       589,824
# │    │    └─BatchNorm3d: 3-89            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-90                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-91                 [1, 1024, 4, 14, 14]      262,144
# │    │    └─BatchNorm3d: 3-92            [1, 1024, 4, 14, 14]      2,048
# │    │    └─ReLU: 3-93                   [1, 1024, 4, 14, 14]      --
# │    └─Bottleneck: 2-11                  [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-94                 [1, 256, 4, 14, 14]       262,144
# │    │    └─BatchNorm3d: 3-95            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-96                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-97                 [1, 256, 4, 14, 14]       589,824
# │    │    └─BatchNorm3d: 3-98            [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-99                   [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-100                [1, 1024, 4, 14, 14]      262,144
# │    │    └─BatchNorm3d: 3-101           [1, 1024, 4, 14, 14]      2,048
# │    │    └─ReLU: 3-102                  [1, 1024, 4, 14, 14]      --
# │    └─Bottleneck: 2-12                  [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-103                [1, 256, 4, 14, 14]       262,144
# │    │    └─BatchNorm3d: 3-104           [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-105                  [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-106                [1, 256, 4, 14, 14]       589,824
# │    │    └─BatchNorm3d: 3-107           [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-108                  [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-109                [1, 1024, 4, 14, 14]      262,144
# │    │    └─BatchNorm3d: 3-110           [1, 1024, 4, 14, 14]      2,048
# │    │    └─ReLU: 3-111                  [1, 1024, 4, 14, 14]      --
# │    └─NLBlockND: 2-13                   [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-112                [1, 512, 4, 14, 14]       524,800
# │    │    └─Conv3d: 3-113                [1, 512, 4, 14, 14]       524,800
# │    │    └─Conv3d: 3-114                [1, 512, 4, 14, 14]       524,800
# │    │    └─Sequential: 3-115            [1, 1024, 4, 14, 14]      527,360
# │    └─Bottleneck: 2-14                  [1, 1024, 4, 14, 14]      --
# │    │    └─Conv3d: 3-116                [1, 256, 4, 14, 14]       262,144
# │    │    └─BatchNorm3d: 3-117           [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-118                  [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-119                [1, 256, 4, 14, 14]       589,824
# │    │    └─BatchNorm3d: 3-120           [1, 256, 4, 14, 14]       512
# │    │    └─ReLU: 3-121                  [1, 256, 4, 14, 14]       --
# │    │    └─Conv3d: 3-122                [1, 1024, 4, 14, 14]      262,144
# │    │    └─BatchNorm3d: 3-123           [1, 1024, 4, 14, 14]      2,048
# │    │    └─ReLU: 3-124                  [1, 1024, 4, 14, 14]      --
# ├─Sequential: 1-9                        [1, 2048, 4, 7, 7]        --
# │    └─Bottleneck: 2-15                  [1, 2048, 4, 7, 7]        --
# │    │    └─Conv3d: 3-125                [1, 512, 4, 14, 14]       524,288
# │    │    └─BatchNorm3d: 3-126           [1, 512, 4, 14, 14]       1,024
# │    │    └─ReLU: 3-127                  [1, 512, 4, 14, 14]       --
# │    │    └─Conv3d: 3-128                [1, 512, 4, 7, 7]         2,359,296
# │    │    └─BatchNorm3d: 3-129           [1, 512, 4, 7, 7]         1,024
# │    │    └─ReLU: 3-130                  [1, 512, 4, 7, 7]         --
# │    │    └─Conv3d: 3-131                [1, 2048, 4, 7, 7]        1,048,576
# │    │    └─BatchNorm3d: 3-132           [1, 2048, 4, 7, 7]        4,096
# │    │    └─Sequential: 3-133            [1, 2048, 4, 7, 7]        2,101,248
# │    │    └─ReLU: 3-134                  [1, 2048, 4, 7, 7]        --
# │    └─Bottleneck: 2-16                  [1, 2048, 4, 7, 7]        --
# │    │    └─Conv3d: 3-135                [1, 512, 4, 7, 7]         1,048,576
# │    │    └─BatchNorm3d: 3-136           [1, 512, 4, 7, 7]         1,024
# │    │    └─ReLU: 3-137                  [1, 512, 4, 7, 7]         --
# │    │    └─Conv3d: 3-138                [1, 512, 4, 7, 7]         2,359,296
# │    │    └─BatchNorm3d: 3-139           [1, 512, 4, 7, 7]         1,024
# │    │    └─ReLU: 3-140                  [1, 512, 4, 7, 7]         --
# │    │    └─Conv3d: 3-141                [1, 2048, 4, 7, 7]        1,048,576
# │    │    └─BatchNorm3d: 3-142           [1, 2048, 4, 7, 7]        4,096
# │    │    └─ReLU: 3-143                  [1, 2048, 4, 7, 7]        --
# │    └─Bottleneck: 2-17                  [1, 2048, 4, 7, 7]        --
# │    │    └─Conv3d: 3-144                [1, 512, 4, 7, 7]         1,048,576
# │    │    └─BatchNorm3d: 3-145           [1, 512, 4, 7, 7]         1,024
# │    │    └─ReLU: 3-146                  [1, 512, 4, 7, 7]         --
# │    │    └─Conv3d: 3-147                [1, 512, 4, 7, 7]         2,359,296
# │    │    └─BatchNorm3d: 3-148           [1, 512, 4, 7, 7]         1,024
# │    │    └─ReLU: 3-149                  [1, 512, 4, 7, 7]         --
# │    │    └─Conv3d: 3-150                [1, 2048, 4, 7, 7]        1,048,576
# │    │    └─BatchNorm3d: 3-151           [1, 2048, 4, 7, 7]        4,096
# │    │    └─ReLU: 3-152                  [1, 2048, 4, 7, 7]        --
# ├─AvgPool3d: 1-10                        [1, 2048, 1, 1, 1]        --
# ├─Linear: 1-11                           [1, 12]                   24,588
# ==========================================================================================
# Total params: 25,634,380
# Trainable params: 25,634,380
# Non-trainable params: 0
# Total mult-adds (G): 21.14
# ==========================================================================================
# Input size (MB): 19.27
# Forward/backward pass size (MB): 1075.25
# Params size (MB): 102.54
# Estimated Total Size (MB): 1197.05
# ==========================================================================================



if __name__ == '__main__':
    import torch
    image = torch.randn(size=(1,3,32,224,224))
    net = resnet3D50(non_local=True)
    summary(model=net, input_size=(1,3,32,224,224))
    # count = 0
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         count += 1
    #         print(name)
    # print (count)
    # out = net(image)
    # print(out.size())   

    # for bn_layer in [True, False]:
        # img = torch.zeros(2, 3, 20)
        # net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
        # out = net(img)
        # print(out.size())

        # img = torch.zeros(2, 3, 20, 20)
        # net = NLBlockND(in_channels=3, mode='concatenate', dimension=2, bn_layer=bn_layer)
        # out = net(img)
        # print(out.size())

        # img = torch.randn(2, 3, 8, 20, 20).to('cuda')
        # net = NLBlockND(in_channels=3, mode='concatenate', dimension=3, bn_layer=bn_layer).to('cuda')
        # summary(model=net,input_size=(2, 3, 4, 20, 20),device='cuda')
        # out = net(img)
        # print(out.size())

