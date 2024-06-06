import torch
import torch.nn as nn
from torchinfo import summary
import math
from torch.nn.modules.utils import _triple
# from conv_2dplus import Conv2DPlus1D
# from corr import WeightedCorrelationBlock
# from build import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv2DPlus1D(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2DPlus1D, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU(inplace=True)

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

class WeightedCorrelationBlock(nn.Module):
    """
    A separate block added at the end of each res-stage in R(2+1)D. 
    See the paper. 
    """
    def __init__(self, num_channel, seq_len, filter_size, dilation=1, num_groups=1, mode="sum"):
        """
        Args:
            see WeightedCorrelationLayer.
        """
        super(WeightedCorrelationBlock, self).__init__()

        assert mode in ["sum", "concat"]
        assert num_channel % 4 == 0

        self.encode_conv = nn.Sequential(
            nn.Conv3d(num_channel, num_channel // 4, 1, bias=False),
            nn.BatchNorm3d(num_channel // 4), 
            nn.ReLU(inplace=True)
        )

        self.correlation = WeightedCorrelationLayer(
            num_channel // 4, seq_len, filter_size, dilation, num_groups
        )
        if mode == "concat":
            assert num_channel > filter_size * filter_size * num_groups
            self.sum_mode = False

            self.bypass_conv = nn.Sequential(
                nn.Conv3d(
                    num_channel // 4, num_channel - filter_size * filter_size * num_groups, 1, bias=False
                ),
                nn.BatchNorm3d(num_channel - filter_size * filter_size * num_groups)
            )

        elif mode == "sum":
            self.sum_mode = True
            self.decode_conv = nn.Sequential(
                nn.Conv3d(filter_size * filter_size * num_groups, num_channel, 1, bias=False),
                nn.BatchNorm3d(num_channel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channel, seq, h, w)
        Returns:
            out (Tensor): shape (batch, channel, seq, h, w)
        """
        x_enc = self.encode_conv(x)
        x_corr = self.correlation(x_enc)

        if self.sum_mode:
            x_dec = self.decode_conv(x_corr)
            out = x + x_dec
        else:
            x_bypass = self.bypass_conv(x_enc)
            out = torch.cat([x_corr, x_bypass], 1)
        out = self.relu(out)
        return out


class WeightedCorrelationLayer(nn.Module):
    """
    Weighted Correlation Layer proposed in paper
    ``Video Modeling with Correlation Networks``. 
    """
    def __init__(self, in_channel, seq_len, filter_size, dilation=1, num_groups=1):
        """
        Args:
            in_channel: C
            seq_len: L
            filter_size: K
            dilation: D. If greater than 1, perform dilated correlation.
            num_groups: G. If greater than 1, perform groupwise correlation.
        """
        super(WeightedCorrelationLayer, self).__init__()

        assert dilation >= 1, "Dilation must be greater than 1. "
        assert num_groups >= 1, "Group number must be greater than 1. "
        assert filter_size % 2 == 1, "Only support odd K. "
        assert in_channel % num_groups == 0, "Group number must be a divisor of channel number. "

        self.filter_weight = nn.Parameter(torch.Tensor(in_channel, seq_len, filter_size, filter_size))
        nn.init.kaiming_normal_(self.filter_weight, mode='fan_out', nonlinearity='relu')

        self.in_channel = in_channel
        self.seq_len = seq_len
        self.dilation = dilation
        self.num_groups = num_groups
        self.span_size = (filter_size - 1) * dilation + 1
        self.pad_size = (self.span_size - 1) // 2


    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channel, seq, h, w)
        Returns:
            flatten_response (Tensor): shape (batch, n_groups*k^2, seq, h, w)
        """

        # second image in each correlation operation
        x2 = F.pad(
            x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            'constant', 0
        )
        # first image in each correlation operation
        # repeat the first frame once to perform self-correlation
        x1 = torch.cat(
            (x[:, :, [0], :, :], x[:, :, :-1, :, :]), 2
        )

        offset_y, offset_x = torch.meshgrid(
            torch.arange(0, self.span_size, self.dilation), 
            torch.arange(0, self.span_size, self.dilation)
        )

        batch_size, c, t, h, w = x.size()
        position_response_list = []

        # for each position in the filter, calculate all responses between two frames
        for dx, dy in zip(offset_x.reshape(-1), offset_y.reshape(-1)):
            pos_filter_weight = self.filter_weight[:, :, dy//self.dilation, dx//self.dilation].view(1, c, t, 1, 1).expand(
                batch_size, -1, -1, h, w
            )
            position_response = pos_filter_weight * x1 * x2[:, :, :, dy:dy+h, dx:dx+w]

            # perform groupwise mean
            position_response = position_response.reshape(
                -1, self.num_groups, c // self.num_groups, t, h, w
            )
            position_response = torch.mean(position_response, 2)

            # position_response: (batch, n_groups, t, h, w)
            position_response_list.append(position_response)

        flatten_response = torch.cat(position_response_list, 1)
        return flatten_response

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size, stride=1):

        super(Bottleneck, self).__init__()

        padding = kernel_size // 2

        self.downsample = (stride > 1) if isinstance(stride, int) else (max(stride) > 1)
        self.stride = stride

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 2+1D Conv
        self.conv2 = nn.Sequential(
            Conv2DPlus1D(planes, planes, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, inplanes, kernel_size=1, bias=False),
            nn.BatchNorm3d(inplanes)
        )
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )
        self.dimchange = nn.Conv3d(in_channels=inplanes,out_channels=planes * self.expansion,stride=1,padding=0,kernel_size=1)
    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample_conv(x)
        
        out = self.dimchange(out)
        
        # print(out.shape, residual.shape)

        out += residual
        out = self.relu(out)

        return out


class CorrNetImpl(nn.Module):

    def __init__(self, layer_sizes, corr_block_locs, num_frames, num_classes=12):
        """
        Args:
            layer_sizes (Seq[int]): number of blocks per layer (including correlation blocks). 
            corr_block_locs (Seq[Seq[int]/None]): 
                indices of correlation blocks in each layer. None if no correlation block is inserted.
            num_classes (int): number of output classes. 
        """
        super(CorrNetImpl, self).__init__()

        self.num_frames = num_frames

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.res2 = self._make_layer(
            32, 64, 3, layer_sizes[0], corr_block_locs[0], stride=(1, 2, 2)
        )
        self.res3 = self._make_layer(
            64 * Bottleneck.expansion, 128, 3, layer_sizes[1], corr_block_locs[1], stride=(1, 2, 2)
        )
        self.res4 = self._make_layer(
            128 * Bottleneck.expansion, 256, 3, layer_sizes[2], corr_block_locs[2], stride=(2, 2, 2)
        )
        self.res5 = self._make_layer(
            256 * Bottleneck.expansion, 512, 3, layer_sizes[3], corr_block_locs[3], stride=(2, 2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._initialize_weights()
        self.flatten = nn.Flatten(start_dim=1,end_dim=4)
        

    def forward(self, x):
        """
        Args:
            x (list): Contains exactly ONE tensor of shape (b, c, t, h, w). 
                      This follows the interface of PySlowFast. 
        Returns:
            x (Tensor): Logits for each class. 
        """
        # x = x[0]
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        # print(f'x:{x.shape}')
        x = self.fc(x)

        return x

    def _make_layer(self, inplanes, planes, kernel_size, layer_size, corr_block_locs, stride=1):
        if corr_block_locs:
            assert min(corr_block_locs) > 0, "Can not insert correlation block at the first location of a layer."
        else:
            corr_block_locs = []
        
        blocks = []

        # first block, need downsampling if stride > 1
        blocks.append(Bottleneck(inplanes, planes, kernel_size, stride))

        # decrease temporal length
        temporal_stride = stride if isinstance(stride, int) else stride[0]
        self.num_frames = self.num_frames // temporal_stride

        inplanes = planes * Bottleneck.expansion
        for i in range(1, layer_size):
            if i in corr_block_locs:
                blocks.append(
                    WeightedCorrelationBlock(inplanes, self.num_frames, filter_size=7, dilation=2, num_groups=32)
                )
            else:
                # no downsampling
                blocks.append(Bottleneck(inplanes, planes, kernel_size))

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class CorrNet(CorrNetImpl):

    def __init__(self, cfg):
        assert cfg.CORR_NET.TYPE in ["corr_26", "corr_50", "corr_101"]
        
        if cfg.CORR_NET.TYPE == "corr_26":
            super(CorrNet, self).__init__(
                [3, 3, 3, 2], [[2], [2], [2], None], cfg.DATA.NUM_FRAMES, cfg.MODEL.NUM_CLASSES
            )
        elif cfg.CORR_NET.TYPE == "corr_50":
            super(CorrNet, self).__init__(
                [4, 5, 7, 3], [[3], [4], [6], None], cfg.DATA.NUM_FRAMES, cfg.MODEL.NUM_CLASSES
            )
        elif cfg.CORR_NET.TYPE == "corr_101":
            super(CorrNet, self).__init__(
                [4, 5, 25, 3], [[3], [4], [12, 24], None], cfg.DATA.NUM_FRAMES, cfg.MODEL.NUM_CLASSES
            )
# shape of corrnet
# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# CorrNetImpl                                        [4, 12]                   --
# ├─Sequential: 1-1                                  [4, 32, 8, 56, 56]        --
# │    └─Conv3d: 2-1                                 [4, 32, 8, 56, 56]        4,736
# │    └─BatchNorm3d: 2-2                            [4, 32, 8, 56, 56]        64
# │    └─ReLU: 2-3                                   [4, 32, 8, 56, 56]        --
# ├─Sequential: 1-2                                  [4, 256, 8, 28, 28]       --
# │    └─Bottleneck: 2-4                             [4, 256, 8, 28, 28]       --
# │    │    └─Sequential: 3-1                        [4, 64, 8, 56, 56]        2,176
# │    │    └─Sequential: 3-2                        [4, 64, 8, 28, 28]        111,216
# │    │    └─Sequential: 3-3                        [4, 32, 8, 28, 28]        2,112
# │    │    └─Sequential: 3-4                        [4, 256, 8, 28, 28]       8,704
# │    │    └─Conv3d: 3-5                            [4, 256, 8, 28, 28]       8,448
# │    │    └─ReLU: 3-6                              [4, 256, 8, 28, 28]       --
# │    └─WeightedCorrelationBlock: 2-5               [4, 256, 8, 28, 28]       --
# │    │    └─Sequential: 3-7                        [4, 64, 8, 28, 28]        16,512
# │    │    └─WeightedCorrelationLayer: 3-8          [4, 1568, 8, 28, 28]      25,088
# │    │    └─Sequential: 3-9                        [4, 256, 8, 28, 28]       401,920
# │    │    └─ReLU: 3-10                             [4, 256, 8, 28, 28]       --
# │    └─Bottleneck: 2-6                             [4, 256, 8, 28, 28]       --
# │    │    └─Sequential: 3-11                       [4, 64, 8, 28, 28]        16,512
# │    │    └─Sequential: 3-12                       [4, 64, 8, 28, 28]        111,216
# │    │    └─Sequential: 3-13                       [4, 256, 8, 28, 28]       16,896
# │    │    └─Conv3d: 3-14                           [4, 256, 8, 28, 28]       65,792
# │    │    └─ReLU: 3-15                             [4, 256, 8, 28, 28]       --
# ├─Sequential: 1-3                                  [4, 512, 8, 14, 14]       --
# │    └─Bottleneck: 2-7                             [4, 512, 8, 14, 14]       --
# │    │    └─Sequential: 3-16                       [4, 128, 8, 28, 28]       33,024
# │    │    └─Sequential: 3-17                       [4, 128, 8, 14, 14]       443,616
# │    │    └─Sequential: 3-18                       [4, 256, 8, 14, 14]       33,280
# │    │    └─Sequential: 3-19                       [4, 512, 8, 14, 14]       132,096
# │    │    └─Conv3d: 3-20                           [4, 512, 8, 14, 14]       131,584
# │    │    └─ReLU: 3-21                             [4, 512, 8, 14, 14]       --
# │    └─WeightedCorrelationBlock: 2-8               [4, 512, 8, 14, 14]       --
# │    │    └─Sequential: 3-22                       [4, 128, 8, 14, 14]       65,792
# │    │    └─WeightedCorrelationLayer: 3-23         [4, 1568, 8, 14, 14]      50,176
# │    │    └─Sequential: 3-24                       [4, 512, 8, 14, 14]       803,840
# │    │    └─ReLU: 3-25                             [4, 512, 8, 14, 14]       --
# │    └─Bottleneck: 2-9                             [4, 512, 8, 14, 14]       --
# │    │    └─Sequential: 3-26                       [4, 128, 8, 14, 14]       65,792
# │    │    └─Sequential: 3-27                       [4, 128, 8, 14, 14]       443,616
# │    │    └─Sequential: 3-28                       [4, 512, 8, 14, 14]       66,560
# │    │    └─Conv3d: 3-29                           [4, 512, 8, 14, 14]       262,656
# │    │    └─ReLU: 3-30                             [4, 512, 8, 14, 14]       --
# ├─Sequential: 1-4                                  [4, 1024, 4, 7, 7]        --
# │    └─Bottleneck: 2-10                            [4, 1024, 4, 7, 7]        --
# │    │    └─Sequential: 3-31                       [4, 256, 8, 14, 14]       131,584
# │    │    └─Sequential: 3-32                       [4, 256, 4, 7, 7]         1,771,968
# │    │    └─Sequential: 3-33                       [4, 512, 4, 7, 7]         132,096
# │    │    └─Sequential: 3-34                       [4, 1024, 4, 7, 7]        526,336
# │    │    └─Conv3d: 3-35                           [4, 1024, 4, 7, 7]        525,312
# │    │    └─ReLU: 3-36                             [4, 1024, 4, 7, 7]        --
# │    └─WeightedCorrelationBlock: 2-11              [4, 1024, 4, 7, 7]        --
# │    │    └─Sequential: 3-37                       [4, 256, 4, 7, 7]         262,656
# │    │    └─WeightedCorrelationLayer: 3-38         [4, 1568, 4, 7, 7]        50,176
# │    │    └─Sequential: 3-39                       [4, 1024, 4, 7, 7]        1,607,680
# │    │    └─ReLU: 3-40                             [4, 1024, 4, 7, 7]        --
# │    └─Bottleneck: 2-12                            [4, 1024, 4, 7, 7]        --
# │    │    └─Sequential: 3-41                       [4, 256, 4, 7, 7]         262,656
# │    │    └─Sequential: 3-42                       [4, 256, 4, 7, 7]         1,771,968
# │    │    └─Sequential: 3-43                       [4, 1024, 4, 7, 7]        264,192
# │    │    └─Conv3d: 3-44                           [4, 1024, 4, 7, 7]        1,049,600
# │    │    └─ReLU: 3-45                             [4, 1024, 4, 7, 7]        --
# ├─Sequential: 1-5                                  [4, 2048, 2, 4, 4]        --
# │    └─Bottleneck: 2-13                            [4, 2048, 2, 4, 4]        --
# │    │    └─Sequential: 3-46                       [4, 512, 4, 7, 7]         525,312
# │    │    └─Sequential: 3-47                       [4, 512, 2, 4, 4]         7,082,880
# │    │    └─Sequential: 3-48                       [4, 1024, 2, 4, 4]        526,336
# │    │    └─Sequential: 3-49                       [4, 2048, 2, 4, 4]        2,101,248
# │    │    └─Conv3d: 3-50                           [4, 2048, 2, 4, 4]        2,099,200
# │    │    └─ReLU: 3-51                             [4, 2048, 2, 4, 4]        --
# │    └─WeightedCorrelationBlock: 2-14              [4, 2048, 2, 4, 4]        --
# │    │    └─Sequential: 3-52                       [4, 512, 2, 4, 4]         1,049,600
# │    │    └─WeightedCorrelationLayer: 3-53         [4, 1568, 2, 4, 4]        50,176
# │    │    └─Sequential: 3-54                       [4, 2048, 2, 4, 4]        3,215,360
# │    │    └─ReLU: 3-55                             [4, 2048, 2, 4, 4]        --
# │    └─Bottleneck: 2-15                            [4, 2048, 2, 4, 4]        --
# │    │    └─Sequential: 3-56                       [4, 512, 2, 4, 4]         1,049,600
# │    │    └─Sequential: 3-57                       [4, 512, 2, 4, 4]         7,082,880
# │    │    └─Sequential: 3-58                       [4, 2048, 2, 4, 4]        1,052,672
# │    │    └─Conv3d: 3-59                           [4, 2048, 2, 4, 4]        4,196,352
# │    │    └─ReLU: 3-60                             [4, 2048, 2, 4, 4]        --
# ├─AdaptiveAvgPool3d: 1-6                           [4, 2048, 1, 1, 1]        --
# ├─Flatten: 1-7                                     [4, 2048]                 --
# ├─Linear: 1-8                                      [4, 12]                   24,588
# ====================================================================================================
# Total params: 41,735,852
# Trainable params: 41,735,852
# Non-trainable params: 0
# Total mult-adds (G): 50.03
# ====================================================================================================
# Input size (MB): 4.82
# Forward/backward pass size (MB): 1747.57
# Params size (MB): 166.94
# Estimated Total Size (MB): 1919.33
# ====================================================================================================
if __name__ == '__main__':
    bottlecck = Bottleneck(inplanes=3, planes=128, kernel_size=1)
    corrnet = CorrNetImpl(layer_sizes=[3,3,3,3],corr_block_locs=[[1],[1],[1],[1]], num_frames=8)
    x = torch.randn(size=(4,3,8,112,112))
    print(corrnet(x).shape)
    # summary(model=corrnet,input_size=(4,3,8,112,112))