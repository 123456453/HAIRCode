import torch
from torch import nn
#是将来自多视角和多模态的数据相互融合，并最终输出分类结果
from RGB.RGB import MultiViewModel, Transformer
from Skeleton.Skele_transformer import SkeletonTransformer
from Skeleton.tool import load_config
from einops import rearrange, reduce
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

#定义全局输出编码器，需要注意的是：多视角模型的输入为：[view_number, batch_size, channels, depths, height, width],多视角模型的输出维度为：[batch_size,3*(nt+1)*nh*nw,embedding_of_globalEncoder ]
#骨骼模型的输入为[batch_size, depths, keypoints, channles], 骨骼模型的输出为：[batch_size, depths+1, embedding_dims]
class Multi_view_modal_model(nn.Module):
    def __init__(self, paras_for_multi_veiw:dict, 
                 paras_for_skeleton:dict, 
                 paras_for_global_encoder:dict,
                 paras_for_overall_encoder:dict,
                 depths:int,
                rank):
        super().__init__()
        self.depths = depths
        self.rank = rank
        self.multiviewmodel = MultiViewModel(fusion_layer=paras_for_multi_veiw['fusion_layer'],
                                             mlp_dims=paras_for_multi_veiw['mlp_dims'],
                                             mlp_dropout=paras_for_multi_veiw['mlp_dropout'],
                                             num_heads=paras_for_multi_veiw['num_heads'],
                                             num_layer=paras_for_multi_veiw['num_layers'],
                                             atten_dropout=paras_for_multi_veiw['atten_dropout'],
                                             embedding_dims=paras_for_multi_veiw['embedding_dims'],
                                             paras_for_global_encoder=paras_for_global_encoder,
                                             view_number=[8,4,2],rank=rank)
        self.skeletontmodel = SkeletonTransformer(mlp_dim=paras_for_skeleton['mlp_dim'],
                                                  mlp_dropout=paras_for_skeleton['mlp_dropout'],
                                                  atten_dropout=paras_for_skeleton['atten_dropout'],
                                                  embedding_dims=paras_for_skeleton['embedding_dims'],
                                                  num_heads=paras_for_skeleton['num_heads'],
                                                  num_layers=paras_for_skeleton['num_layers'],
                                                 rank=rank)
        self.overalltransformer = Transformer(atten_dropout=paras_for_overall_encoder['atten_dropout'],
                                       num_heads=paras_for_overall_encoder['num_heads'],
                                       mlp_dim=paras_for_overall_encoder['mlp_dim'],
                                       mlp_dropout=paras_for_overall_encoder['mlp_dropout'],
                                       embedding_dims=paras_for_overall_encoder['embedding_dims'],
                                             rank=rank)
    def forward(self, x_for_multiview, x_for_skeleton):
        #注意multiview的全局编码器和骨架模型的输出编码器的维度要保持一致，否则需要额外的维度转换
        #这里的输入是来自多视角和骨骼模型的输出的融合，至于具体怎么融合，需要考虑一下
        # print(f'MMM:{x_for_multiview[0].shape},{x_for_multiview[1].shape}')
        y_from_multiview = self.multiviewmodel(x_for_multiview) #torch.Size([2, 5292, 1024]
        y_from_skeleton = self.skeletontmodel(x_for_skeleton) #torch.Size([2, 33, 1024]
        fusion_input = torch.cat((y_from_multiview,y_from_skeleton),dim=1) #torch.Size([2, 5325, 1024]
        out = self.overalltransformer(fusion_input)
        #进行维度转换
        out = rearrange(out, 'b t d->b d t')
        #在时间维度拉到与输入depth持平
        out = nn.Linear(in_features=out.shape[-1], out_features=self.depths).to(self.rank)(out)
        #拉回
        out = rearrange(out, 'b d t->b t d')
        out = nn.Linear(in_features=out.shape[-1], out_features=12).to(self.rank)(out)
        #输出的维度为[batch_size, depths, classifier_number],其中classifier_number为每一个视频钟对应帧数的labels
        return out
