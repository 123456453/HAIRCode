# \参考自：Quanfu Fan, Chun-Fu Chen, Hilde Kuehne, Marco Pistoia,and David Cox. More is less: Learning efficient video repre-sentations by big-little network and depthwise temporal aggregation. In NeurIPS, 2019.
#导入必要的包

import itertools
from collections import OrderedDict

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchinfo import summary
__all__ = ['bLVNet_TAM_BACKBONE', 'blvnet_tam_backbone']

model_urls = {
    'blresnet50': 'pretrained/ImageNet-bLResNet-50-a2-b4.pth.tar',
    'blresnet101': 'pretrained/ImageNet-bLResNet-101-a2-b4.pth.tar'
}
class TAM(nn.Module):

    def __init__(self, duration, channels, blending_frames=3):
        super().__init__()
        self.blending_frames = blending_frames
        self.channels = channels

        if blending_frames == 3:
            self.prev = nn.Conv2d(channels, channels, kernel_size=1,
                                     padding=0, groups=channels, bias=False)
            self.next = nn.Conv2d(channels, channels, kernel_size=1,
                                     padding=0, groups=channels, bias=False)
            self.curr = nn.Conv2d(channels, channels, kernel_size=1,
                                     padding=0, groups=channels, bias=False)
        else:
            self.blending_layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1,
                                                            padding=0, groups=channels, bias=False)
                                                  for i in range(blending_frames)])
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration

    def forward(self, x):
        #输入维度[batch_size, channel, height, width],输出维度：[batch_size, channel, height, width]
        if self.blending_frames == 3:

            prev_x = self.prev(x) #torch.Size([8, 3, 112, 112])
    
            curr_x = self.curr(x) #torch.Size([8, 3, 112, 112])
            next_x = self.next(x) #torch.Size([8, 3, 112, 112])
            #在这里将[batch_size, channels, height, width]提升一个维度，变为[n, duration, channels, height, width],值得注意是:n*duration=batch_size
            prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:]) #torch.Size([8, 1, 3, 112, 112])
            curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:]) #torch.Size([8, 1, 3, 112, 112])
            next_x = next_x.view((-1, self.duration) + next_x.size()[1:]) #torch.Size([8, 1, 3, 112, 112])
            prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...] #torch.Size([8, 1, 3, 112, 112])
            next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...] #torch.Size([8, 1, 3, 112, 112])
            out = torch.stack([prev_x, curr_x, next_x], dim=0) ##torch.Size([8, 1, 3, 112, 112])
        else:
            # multiple blending
            xs = [se(x) for se in self.blending_layers]
            xs = [x.view((-1, self.duration) + x.size()[1:]) for x in xs]

            shifted_xs = []
            for i in range(self.blending_frames):
                shift = i - (self.blending_frames // 2)
                x_temp = xs[i]
                n, t, c, h, w = x_temp.shape
                start_index = 0 if shift < 0 else shift
                end_index = t if shift < 0 else t + shift
                padding = None
                if shift < 0:
                    padding = (0, 0, 0, 0, 0, 0, abs(shift), 0)
                elif shift > 0:
                    padding = (0, 0, 0, 0, 0, 0, 0, shift)
                shifted_xs.append(F.pad(x_temp, padding)[:, start_index:end_index, ...]
                                  if padding is not None else x_temp)

            out = torch.stack(shifted_xs, dim=0)
        out = torch.sum(out, dim=0) #torch.Size([8, 1, 3, 112, 112])
        out = self.relu(out)
        # [N, T, C, N, H]
        out = out.view((-1, ) + out.size()[2:]) ##torch.Size([8, 1, 3, 112, 112])
        return out


def get_frame_list(init_list, num_frames, batch_size):
    if batch_size == 0:
        return []

    flist = list()
    for i in range(batch_size):
        flist.append([k + i * num_frames for k in init_list])
    return list(itertools.chain(*flist))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True,
                 with_tam=False, num_frames=-1, blending_frames=-1):

        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        self.last_relu = last_relu

        self.tam = TAM(num_frames, inplanes, blending_frames) \
            if with_tam else None

    def forward(self, x):
         #输入维度[batch_size, channel, height, width],输出维度：[batch_size, channel, height, width]
        residual = x  #torch.Size([8, 3, 112, 112])
        if self.tam is not None:
            x = self.tam(x)
        out = self.conv1(x) #torch.Size([8, 1, 112, 112])

        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out) ##torch.Size([8, 1, 112, 112])
 
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out) #torch.Size([8, 3, 112, 112])
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out #torch.Size([8, 3, 112, 112])
class bLModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, alpha, beta, stride,
                 num_frames, blending_frames=3):
        super(bLModule, self).__init__()
        self.num_frames = num_frames
        self.blending_frames = blending_frames

        self.relu = nn.ReLU(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1, 1, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha, max(1, blocks // beta - 1))
        self.little_e = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels))

        self.fusion = self._make_layer(block, in_channels, out_channels, 1, stride=stride)
        self.tam = TAM(self.num_frames, in_channels, blending_frames=self.blending_frames)
    #make_layer的目的是制造更多的残差网络
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=1, padding=1))
        if inplanes != planes:

            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, stride, downsample))
        else:
            layers.append(block(inplanes, planes, stride, downsample))

 
            for i in range(1, blocks):
                layers.append(block(inplanes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x, big_frame_num, big_list, little_frame_num, little_list):
        #输入维度为[batch_size, in_channels, height, width]
        n = x.size()[0]

        if self.tam is not None:
            x = self.tam(x)
        big = self.big(x[big_list, ::])

        little = self.little(x[little_list, ::])

        little = self.little_e(little)
    
        #torch.nn.functional.interpolate是一个对输入进行上采样或者下采样的函数
        big = torch.nn.functional.interpolate(big, little.shape[2:])

        # [0 1] sum up current and next frames
        bn = big_frame_num

        ln = little_frame_num


        big = big.view((-1, bn) + big.size()[1:])

        little = little.view((-1, ln) + little.size()[1:])

        big += little  # left frame

        # only do the big branch
        big = big.view((-1,) + big.size()[2:])
        big = self.relu(big)
        big = self.fusion(big)


        # distribute big to both
        x = torch.zeros((n,) + big.size()[1:], device=big.device, dtype=big.dtype)

        x[range(0, n, 2), ::] = big

        x[range(1, n, 2), ::] = big


        return x #torch.Size([8, 3, 112, 112])


class bLVNet_TAM_BACKBONE(nn.Module):

    def __init__(self, block, layers, alpha, beta, num_frames, num_classes=12,
                 blending_frames=3, input_channels=3):

        self.num_frames = num_frames
        self.blending_frames = blending_frames

        self.bL_ratio = 2
        self.big_list = list(range(self.bL_ratio // 2, num_frames, self.bL_ratio))
        self.little_list = list(set(range(0, num_frames)) - set(self.big_list))

        num_channels = [64, 128, 256, 512]
        self.inplanes = 64

        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2,
                                 padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])

        self.tam = TAM(self.num_frames, num_channels[0], blending_frames=self.blending_frames)

        self.layer1 = bLModule(block, num_channels[0], num_channels[0] * block.expansion,
                               layers[0], alpha, beta, stride=1, num_frames=self.num_frames,
                               blending_frames=blending_frames)
        self.layer2 = bLModule(block, num_channels[0] * block.expansion,
                               num_channels[1] * block.expansion, layers[1], alpha, beta, stride=1,
                               num_frames=self.num_frames,
                               blending_frames=blending_frames)
        self.layer2conv = nn.Conv2d(in_channels=64, out_channels=num_channels[0] * block.expansion, stride=1,padding=1,kernel_size=3)
        self.layer3 = bLModule(block, num_channels[1] * block.expansion,
                               num_channels[2] * block.expansion, layers[2], alpha, beta, stride=1,
                               num_frames=self.num_frames,
                               blending_frames=blending_frames)
        self.layer3conv = nn.Conv2d(in_channels=256, out_channels=num_channels[1] * block.expansion,stride=1,padding=1,kernel_size=3)
        # only half frames are used.
        self.layer4 = self._make_layer(
            block, num_channels[2] * block.expansion, num_channels[3] * block.expansion, layers[3],
            num_frames=self.num_frames // 2, stride=1)
        self.layer4conv = nn.Conv2d(in_channels=512, out_channels=num_channels[2] * block.expansion,stride=1,padding=1,kernel_size=3)

        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, num_frames, stride=1, with_tam=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, with_tam=with_tam,
                            num_frames=num_frames, blending_frames=self.blending_frames))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, with_tam=with_tam,
                                num_frames=num_frames, blending_frames=self.blending_frames))

        return nn.Sequential(*layers)

    def _forward_bL_layer0(self, x, big_frame_num, big_list, little_frame_num, little_list):
        n = x.size()[0]
        if self.tam is not None:
            x = self.tam(x)

        bx = self.b_conv0(x[big_list, ::])
        bx = self.bn_b0(bx)

        lx = self.l_conv0(x[little_list, ::])
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)

        bn = big_frame_num
        ln = little_frame_num
        bx = bx.view((-1, bn) + bx.size()[1:])
        lx = lx.view((-1, ln) + lx.size()[1:])
        bx += lx   # left frame

        bx = bx.view((-1,) + bx.size()[2:])

        bx = self.relu(bx)
        bx = self.bl_init(bx)
        bx = self.bn_bl_init(bx)
        bx = self.relu(bx)

        x = torch.zeros((n,) + bx.size()[1:], device=bx.device, dtype=bx.dtype)
        x[range(0, n, 2), ::] = bx
        x[range(1, n, 2), ::] = bx

        return x

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        n = x.size()[0]

        batch_size = n // self.num_frames

        big_list = get_frame_list(self.big_list, self.num_frames, batch_size)

        little_list = get_frame_list(self.little_list, self.num_frames, batch_size)


        x = self._forward_bL_layer0(x, len(self.big_list), big_list, len(self.little_list), little_list)  #torch.Size([8, 64, 28, 28])

        x = self.layer1(x, len(self.big_list), big_list, len(self.little_list), little_list) #torch.Size([8, 64, 28, 28])

        x = self.layer2conv(x)

        x = self.layer2(x, len(self.big_list), big_list, len(self.little_list), little_list)

        x = self.layer3conv(x)

        x = self.layer3(x, len(self.big_list), big_list, len(self.little_list), little_list)

        x = self.layer4conv(x)

        x = self.layer4(x[big_list, ::])
        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def blvnet_tam_backbone(depth, alpha, beta, num_frames, blending_frames=3, input_channels=3,
                        imagenet_blnet_pretrained=True):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]

    model = bLVNet_TAM_BACKBONE(Bottleneck, layers, alpha, beta, num_frames,
                                blending_frames=blending_frames, input_channels=input_channels)

    if imagenet_blnet_pretrained:
        checkpoint = torch.load(model_urls['blresnet{}'.format(depth)], map_location='cpu')
        print("loading weights from ImageNet-pretrained blnet, blresnet{}".format(depth),
              flush=True)
        # fixed parameter names in order to load the weights correctly
        state_d = OrderedDict()
        if input_channels != 3:  # flow
            print("Convert RGB model to Flow")
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('module.', '')
                if "conv1.weight" in key:
                    o_c, in_c, k_h, k_w = value.shape
                else:
                    o_c, in_c, k_h, k_w = 0, 0, 0, 0
                if k_h == 7 and k_w == 7:
                    # average the weights and expand to all channels
                    new_shape = (o_c, input_channels, k_h, k_w)
                    new_value = value.mean(dim=1, keepdim=True).expand(new_shape).contiguous()
                else:
                    new_value = value
                state_d[new_key] = new_value
        else:
            print("Loading RGB model")
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('module.', '')                
                state_d[new_key] = value
        msg = model.load_state_dict(state_d, strict=False)
        print(msg, flush=True)

    return model

#shape of blvnet:
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# bLVNet_TAM_BACKBONE                      [4, 12]                   --
# ├─Conv2d: 1-1                            [8, 64, 56, 56]           9,408
# ├─BatchNorm2d: 1-2                       [8, 64, 56, 56]           128
# ├─ReLU: 1-3                              [8, 64, 56, 56]           --
# ├─TAM: 1-4                               [8, 64, 56, 56]           --
# │    └─Conv2d: 2-1                       [8, 64, 56, 56]           64
# │    └─Conv2d: 2-2                       [8, 64, 56, 56]           64
# │    └─Conv2d: 2-3                       [8, 64, 56, 56]           64
# │    └─ReLU: 2-4                         [4, 2, 64, 56, 56]        --
# ├─Conv2d: 1-5                            [4, 64, 28, 28]           36,864
# ├─BatchNorm2d: 1-6                       [4, 64, 28, 28]           128
# ├─Conv2d: 1-7                            [4, 64, 56, 56]           36,864
# ├─BatchNorm2d: 1-8                       [4, 64, 56, 56]           128
# ├─ReLU: 1-9                              [4, 64, 56, 56]           --
# ├─Conv2d: 1-10                           [4, 64, 28, 28]           36,864
# ├─BatchNorm2d: 1-11                      [4, 64, 28, 28]           128
# ├─ReLU: 1-12                             [4, 64, 28, 28]           --
# ├─Conv2d: 1-13                           [4, 64, 28, 28]           4,096
# ├─BatchNorm2d: 1-14                      [4, 64, 28, 28]           128
# ├─ReLU: 1-15                             [4, 64, 28, 28]           --
# ├─Conv2d: 1-16                           [4, 64, 28, 28]           4,096
# ├─BatchNorm2d: 1-17                      [4, 64, 28, 28]           128
# ├─ReLU: 1-18                             [4, 64, 28, 28]           --
# ├─bLModule: 1-19                         [8, 64, 28, 28]           --
# │    └─TAM: 2-5                          [8, 64, 28, 28]           --
# │    │    └─Conv2d: 3-1                  [8, 64, 28, 28]           64
# │    │    └─Conv2d: 3-2                  [8, 64, 28, 28]           64
# │    │    └─Conv2d: 3-3                  [8, 64, 28, 28]           64
# │    │    └─ReLU: 3-4                    [4, 2, 64, 28, 28]        --
# │    └─Sequential: 2-6                   [4, 64, 28, 28]           --
# │    │    └─Bottleneck: 3-5              [4, 64, 28, 28]           45,440
# │    │    └─Bottleneck: 3-6              [4, 64, 28, 28]           45,440
# │    └─Sequential: 2-7                   [4, 64, 28, 28]           --
# │    │    └─Bottleneck: 3-7              [4, 64, 28, 28]           45,440
# │    │    └─Bottleneck: 3-8              [4, 64, 28, 28]           45,440
# │    └─Sequential: 2-8                   [4, 64, 28, 28]           --
# │    │    └─Conv2d: 3-9                  [4, 64, 28, 28]           4,096
# │    │    └─BatchNorm2d: 3-10            [4, 64, 28, 28]           128
# │    └─ReLU: 2-9                         [4, 64, 28, 28]           --
# │    └─Sequential: 2-10                  [4, 64, 28, 28]           --
# │    │    └─Bottleneck: 3-11             [4, 64, 28, 28]           45,440
# ├─Conv2d: 1-20                           [8, 256, 28, 28]          147,712
# ├─bLModule: 1-21                         [8, 256, 28, 28]          --
# │    └─TAM: 2-11                         [8, 256, 28, 28]          --
# │    │    └─Conv2d: 3-12                 [8, 256, 28, 28]          256
# │    │    └─Conv2d: 3-13                 [8, 256, 28, 28]          256
# │    │    └─Conv2d: 3-14                 [8, 256, 28, 28]          256
# │    │    └─ReLU: 3-15                   [4, 2, 256, 28, 28]       --
# │    └─Sequential: 2-12                  [4, 256, 28, 28]          --
# │    │    └─Bottleneck: 3-16             [4, 256, 28, 28]          214,016
# │    │    └─Bottleneck: 3-17             [4, 256, 28, 28]          214,016
# │    │    └─Bottleneck: 3-18             [4, 256, 28, 28]          214,016
# │    └─Sequential: 2-13                  [4, 256, 28, 28]          --
# │    │    └─Bottleneck: 3-19             [4, 256, 28, 28]          214,016
# │    │    └─Bottleneck: 3-20             [4, 256, 28, 28]          214,016
# │    │    └─Bottleneck: 3-21             [4, 256, 28, 28]          214,016
# │    └─Sequential: 2-14                  [4, 256, 28, 28]          --
# │    │    └─Conv2d: 3-22                 [4, 256, 28, 28]          65,536
# │    │    └─BatchNorm2d: 3-23            [4, 256, 28, 28]          512
# │    └─ReLU: 2-15                        [4, 256, 28, 28]          --
# │    └─Sequential: 2-16                  [4, 256, 28, 28]          --
# │    │    └─Bottleneck: 3-24             [4, 256, 28, 28]          214,016
# ├─Conv2d: 1-22                           [8, 512, 28, 28]          1,180,160
# ├─bLModule: 1-23                         [8, 512, 28, 28]          --
# │    └─TAM: 2-17                         [8, 512, 28, 28]          --
# │    │    └─Conv2d: 3-25                 [8, 512, 28, 28]          512
# │    │    └─Conv2d: 3-26                 [8, 512, 28, 28]          512
# │    │    └─Conv2d: 3-27                 [8, 512, 28, 28]          512
# │    │    └─ReLU: 3-28                   [4, 2, 512, 28, 28]       --
# │    └─Sequential: 2-18                  [4, 512, 28, 28]          --
# │    │    └─Bottleneck: 3-29             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-30             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-31             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-32             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-33             [4, 512, 28, 28]          854,016
# │    └─Sequential: 2-19                  [4, 512, 28, 28]          --
# │    │    └─Bottleneck: 3-34             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-35             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-36             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-37             [4, 512, 28, 28]          854,016
# │    │    └─Bottleneck: 3-38             [4, 512, 28, 28]          854,016
# │    └─Sequential: 2-20                  [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-39                 [4, 512, 28, 28]          262,144
# │    │    └─BatchNorm2d: 3-40            [4, 512, 28, 28]          1,024
# │    └─ReLU: 2-21                        [4, 512, 28, 28]          --
# │    └─Sequential: 2-22                  [4, 512, 28, 28]          --
# │    │    └─Bottleneck: 3-41             [4, 512, 28, 28]          854,016
# ├─Conv2d: 1-24                           [8, 1024, 28, 28]         4,719,616
# ├─Sequential: 1-25                       [4, 1024, 28, 28]         --
# │    └─Bottleneck: 2-23                  [4, 1024, 28, 28]         --
# │    │    └─TAM: 3-42                    [4, 1024, 28, 28]         3,072
# │    │    └─Conv2d: 3-43                 [4, 512, 28, 28]          524,288
# │    │    └─BatchNorm2d: 3-44            [4, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-45                   [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-46                 [4, 512, 28, 28]          2,359,296
# │    │    └─BatchNorm2d: 3-47            [4, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-48                   [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-49                 [4, 1024, 28, 28]         524,288
# │    │    └─BatchNorm2d: 3-50            [4, 1024, 28, 28]         2,048
# │    │    └─ReLU: 3-51                   [4, 1024, 28, 28]         --
# │    └─Bottleneck: 2-24                  [4, 1024, 28, 28]         --
# │    │    └─TAM: 3-52                    [4, 1024, 28, 28]         3,072
# │    │    └─Conv2d: 3-53                 [4, 512, 28, 28]          524,288
# │    │    └─BatchNorm2d: 3-54            [4, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-55                   [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-56                 [4, 512, 28, 28]          2,359,296
# │    │    └─BatchNorm2d: 3-57            [4, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-58                   [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-59                 [4, 1024, 28, 28]         524,288
# │    │    └─BatchNorm2d: 3-60            [4, 1024, 28, 28]         2,048
# │    │    └─ReLU: 3-61                   [4, 1024, 28, 28]         --
# │    └─Bottleneck: 2-25                  [4, 1024, 28, 28]         --
# │    │    └─TAM: 3-62                    [4, 1024, 28, 28]         3,072
# │    │    └─Conv2d: 3-63                 [4, 512, 28, 28]          524,288
# │    │    └─BatchNorm2d: 3-64            [4, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-65                   [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-66                 [4, 512, 28, 28]          2,359,296
# │    │    └─BatchNorm2d: 3-67            [4, 512, 28, 28]          1,024
# │    │    └─ReLU: 3-68                   [4, 512, 28, 28]          --
# │    │    └─Conv2d: 3-69                 [4, 1024, 28, 28]         524,288
# │    │    └─BatchNorm2d: 3-70            [4, 1024, 28, 28]         2,048
# │    │    └─ReLU: 3-71                   [4, 1024, 28, 28]         --
# ├─AdaptiveAvgPool2d: 1-26                [4, 1024, 1, 1]           --
# ├─Linear: 1-27                           [4, 12]                   12,300
# ==========================================================================================
# Total params: 27,889,484
# Trainable params: 27,889,484
# Non-trainable params: 0
# Total mult-adds (G): 106.81
# ==========================================================================================
# Input size (MB): 1.20
# Forward/backward pass size (MB): 1679.49
# Params size (MB): 111.56
# Estimated Total Size (MB): 1792.25
# ==========================================================================================
if __name__ == '__main__':
    input_tam = torch.randn(size=(8,3,112,112))
    tam = TAM(duration=2,channels=3,blending_frames=3)
    bottlenneck = Bottleneck(inplanes=3, planes=8)
    blmoudle = bLModule(block=Bottleneck,in_channels=3,out_channels=8,blocks=3,alpha=2,beta=2,stride=1,num_frames=1)
    blvnet = bLVNet_TAM_BACKBONE(block=Bottleneck, layers=[3, 4, 6, 3], alpha=1, beta=1,num_frames=2)
    # print(bottlenneck(input_tam).shape)
    # print(bottlenneck(input_tam))
    # print((blvnet(input_tam)).shape)
    summary(model=blvnet, input_size=(8,3,112,112))