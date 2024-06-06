import torch 
from torch import nn
import numpy as np
import torch.multiprocessing.spawn
import torch.utils
import torch.utils.data
from train_loop import train
# from Multiview_Multimodal_model import Multi_view_modal_model
# #是将来自多视角和多模态的数据相互融合，并最终输出分类结果
# from RGB.engine import MultiViewModel, Transformer
# from Skeleton.Skele_transformer import SkeletonTransformer
from RGB.tool import load_config
from einops import rearrange, reduce
from Datasets_DataLoader.RGB_Skelelton_datasets import MyRGBSkeletonDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel 
import os
#导入对比算法模型
from Comprasion_alogorithm import ActionTransformer, blVNet, CorrNet, I3DNL, ip_CSN_152, MVIT, ResNet_RNN,RestNet3D_18, SlowFastR101_NL, VIVIT
def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device =  'cpu'
    paras_for_multi_view = load_config('MultiViewModel')
    paras_for_skeleton = load_config('SkeletonModel')
    paras_for_global_encoder = load_config('MultiViewModel')['paras_for_global_encoder']
    paras_for_overall_encoder = load_config('Overclasificationencoder')
    # 启动分布式训练环境
    train_dataset = MyRGBSkeletonDataset(data_root_path='F:/code/SelfAttention_ActionRecognition/Input_dataset')
    print(f'train_dataset = {len(train_dataset)}')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=load_config('Batch_size'),
                                  num_workers=2,
                                  pin_memory=True,
                                  shuffle=True,
                                  collate_fn=None,
                                  drop_last=True)
    # model = CorrNet.CorrNetImpl(layer_sizes=[4,5,3,4],corr_block_locs=[[1],[1],[1],[1]], num_frames=24).to(device)
    model = ResNet_RNN.Res_RNN(block=ResNet_RNN.Bottleneck, blocks_layer_num=4,inplane=3,plane=512,num_class=12,lstm_hiddensize=512).to(device)
    train_loss, train_acc = train(model=model,
                                  train_dataloader=train_dataloader,
                                  Epoches=1500,
                                  device=device)
    

if __name__ == '__main__':
    main()