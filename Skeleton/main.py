import torch 
from torch import nn
import numpy as np
import torch.utils
import torch.utils.data
#是将来自多视角和多模态的数据相互融合，并最终输出分类结果
from tool import load_config
from Skeleton_dataloader import MySkeletonDataset
from torch.utils.data import Dataset, DataLoader
from Skele_transformer import ActionTransformer
from Train_loop import train
def main():
    paras_for_skeleton = load_config('SkeletonModel')
    # 启动分布式训练环境
    train_dataset = MySkeletonDataset(data_root_path='E:/mdx/SelfAttention_ActionRecognition/Input_dataset')
    print(f'train_dataset = {len(train_dataset)}')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=load_config('Batch_size'),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=True,
                                  collate_fn=None,drop_last=True)
    model = ActionTransformer(d_model=paras_for_skeleton['embedding_dims'],
                             d_ff=paras_for_skeleton['mlp_dims'],
                             seq_length=paras_for_skeleton['depths'],
                             n_head=paras_for_skeleton['num_heads'],
                             dropout=0.2,
                             keypoints=paras_for_skeleton['keypoints'],
                             channels=paras_for_skeleton['channels'],
                             encoder_layer=paras_for_skeleton['num_layers']).to('cuda')
    train_loss, train_acc = train(model=model,
                                  train_dataloader=train_dataloader,
                                  Epoches=730)

if __name__ == '__main__':
    main()