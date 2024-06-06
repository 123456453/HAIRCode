#创建transformer模型，进行模型训练

import torch
from torch import nn
import numpy as np
from torchinfo import summary
from einops import rearrange
#首先定义QKV相乘计算注意力值的函数
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def scale_dot_Product_attention(query, key, value, mask=None, dropout=None):
    '''计算QKV的attention的值并返回attention'''
    #d_k等于qk维度的最后一个纬度值，即dmodel
    d_k = query.shape[-1]
    #计算QK乘积
    scores = torch.matmul(query,key.transpose(-1,-2).contiguous())
    #计算根号d_k
    d_k_sqrt = torch.math.sqrt(d_k)
    #计算QK/根号d_k
    scores = scores /d_k_sqrt
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    p_atten = scores.softmax(dim = -1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten

#前向神经网络，用于将特征进行投影映射
class feed_forward_network(nn.Module):
    def __init__(self,d_model, d_ff):
        super(feed_forward_network,self).__init__()
        self.ff11 = nn.Linear(in_features=d_model,out_features=d_ff).to(device)
        self.ff21 = nn.Linear(in_features=d_ff, out_features=d_model).to(device)
        self.gelu = nn.GELU()   #Leaky ReLU
    def forward(self,x):
        #x.shape = [batch_size, seq_length, d_model]
        x = self.ff11(x)
        x = self.gelu(x)
        x = self.ff21(x)
        return x


class multiheadattention(nn.Module):
    def __init__(self,d_model, n_heads,mask=None):
        super(multiheadattention,self).__init__()
        self.d_model = d_model
        self.n_head = n_heads
        self.mask = mask
        self.d_k = d_model // n_heads
        self.fc= nn.Linear(in_features=d_model,out_features=d_model).to(device)
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model).to(device)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model).to(device)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model).to(device)
    def split_heads(self, x, batch_sizes):
        x = torch.reshape(x,[batch_sizes, -1, self.n_head, self.d_k])
        return torch.transpose(x,1,2)
    def forward(self, input_Q, input_K, input_V):
        # input_Q, input_K, input_V = [batch_size, seq_length, d_model]
        batch_size = input_Q.shape[0]
        #Q,K,V的维度变化过程：[batch_size, seq_length, d_model] -> [batch_size, seq_length, n_heads, d_k]
        #交换维度的目的是匹配之前的scale_dot_product函数的点积运算[batch_size, seq_length, d_model]
        q = self.W_Q (input_Q)
        k = self.W_K(input_K)
        v = self.W_V(input_V)
        # [batch_size, seq_length, n_heads, d_k] -> [batch_size, n_head, seq_length, d_k]
        Q = self.split_heads(q,batch_size).to(device)
        K = self.split_heads(k,batch_size).to(device)
        V = self.split_heads(v,batch_size).to(device)

        prob, attention = scale_dot_Product_attention(query=Q, key=K, value=V, mask=self.mask)

        #交换维度[batch_size, n_head, seq_length, d_k] -> [batch_size, seq_length, n_heads, d_k]

        prob = prob.permute(0,2,1,3).contiguous()
        #[batch_size, seq_length, n_heads, d_k] -> [batch_size, seq_length, d_model] concat操作
        prob = prob.view(batch_size, -1, self.n_head*self.d_k).contiguous()

        output = self.fc(prob)
        #此处没有连接残差和layernorm
        return output
#编写transformerencoder
class transformerencoderlayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout,  mask=None):
        super(transformerencoderlayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_k = d_model // n_heads
        self.mask = mask
        self.fc = feed_forward_network(d_model = self.d_model, d_ff = d_ff)
        self.mutiheadattention = multiheadattention(d_model=self.d_model, n_heads=self.n_heads,mask=self.mask)
        self.dropout_operaction_1 = nn.Dropout(p=dropout)
        self.dropout_operaction_2 = nn.Dropout(p=dropout)
        self.layernorm_1 = nn.LayerNorm(d_model,eps=1e-6).to(device)
        self.layernorm_2=nn.LayerNorm(d_model,eps=1e-6).to(device)

    def forward(self, x):
        residual_1 =  x.to(device)
        #经过多头自注意力层
        atten_prob = self.mutiheadattention(x,x,x)
        dropout_1 = self.dropout_operaction_1(atten_prob)
        atten_out = self.layernorm_1(residual_1.to(device) + dropout_1).to(device)

        #经过FF
        residual_2 = atten_out

        FF_out = self.fc(atten_out)
        dropout_2 = self.dropout_operaction_2(FF_out)
        decoder_out = self.layernorm_2(dropout_2.to(device) + residual_2.to(device)).to(device)

        return decoder_out
#多层decoerlayer的结合构成了decoer
class encoder(nn.Module):
    def __init__(self, n_layer, d_model, d_ff, n_heads, dropout,  mask):
        super(encoder,self).__init__()
        self.n_layer = n_layer
        self.n_head = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.encoder_layers = nn.ModuleList([transformerencoderlayer(d_model = self.d_model,
                                                    d_ff = self.d_ff,
                                                    n_heads = self.n_head,
                                                    dropout = self.dropout,
                                                    mask = mask) for i in range(self.n_layer)])
    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x
#class_embedding + position_encoding
class class_position_embedding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(class_position_embedding,self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        #总共有31帧的数据，将每一帧数据映射为一个64维度的向量编码（位置编码） (1,31,64)
        self.position_embedding  = nn.Embedding(num_embeddings=self.seq_length+1,embedding_dim=self.d_model).to(device)
        # class token
        self.class_token = nn.Parameter(torch.rand(size=(256,1,self.d_model))).to(device)
    def forward(self,x):
        #concat class_token (1,1,64) and x (x, 30,64)
        position = torch.LongTensor([[x for x in range(0, 33)]]).to(device)
        x = torch.cat((self.class_token, x),dim=1)  # x.shape = (31,64)
        #实例化positon_encoding
        x = x + self.position_embedding(position)

        return x


class ActionTransformer(nn.Module):
    def __init__(self, d_model, d_ff, seq_length, n_head, dropout, keypoints, channels, encoder_layer, mask=None):
        super(ActionTransformer,self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.keypoints = keypoints
        self.channels = channels
        self.seq_length = seq_length
        self.n_head = n_head
        self.dropout = dropout
        self.encoder_layer = encoder_layer
        self.mask = mask
        self.class_position_embedding = class_position_embedding(d_model=self.d_model, seq_length=self.seq_length)
        self.transformerencoder = encoder(n_layer=self.encoder_layer, d_model=self.d_model,d_ff=self.d_ff,
                                          n_heads=self.n_head,dropout=self.dropout,mask=self.mask)
        self.ff = nn.Linear(in_features=self.keypoints*self.channels, out_features=self.d_model)
        self.ff1 = nn.Linear(in_features=self.d_model, out_features=256)
        self.ff2 = nn.Linear(in_features=256, out_features=12)
        self.tanh = nn.Tanh()
    def forward(self,x):
        #首先将输入维度为[batch_size, seq_length, keypoints*channels]维度的矩阵经过线性层进行投影到维度为[batch_size, seq_length,d_model]
        x = rearrange(x, 'b t k c -> b t (k c)')
        x = self.ff(x) #x.shape = [512,30,64]
        #进行class embedding和position embedding
        x = self.class_position_embedding(x)
        #经过transformer encoding
        x = self.transformerencoder(x)
        #ONLY THE Xcls IS FED INTO MLP
        x = rearrange(x, ' b t d -> b d t')
        x = nn.Linear(in_features=x.shape[-1], out_features=32).to(device)(x)
        x = rearrange(x, 'b d t -> b t d')
        x = self.ff1(x)
        x = self.tanh(x)
        x = self.ff2(x)
        return x


# if __name__ == '__main__':
#     model = ActionTransformer(d_model=192,
#                               d_ff=512,
#                               seq_length=32,
#                               n_head=4,
#                               dropout=0.2,
#                               keypoints=23,
#                               channels=7,
#                               encoder_layer=4).to('cuda')
#     models = torch.load('SuperComputerFile/Action_model_weights_4Layer_04.pth')
#     trained_layer = list(models)
#     untrained_layer = []
#     for key, value in model.state_dict().items():
#         untrained_layer.append(key)
#     del untrained_layer[0]
#     for i in range(-6,0):
#         del untrained_layer[i]
#     print(untrained_layer == trained_layer)
#     for layer in untrained_layer:
#         to_load = {k:v for  k, v in models.items() if layer in k}
#         model.load_state_dict(to_load, strict=False)
#     print(model.state_dict())
