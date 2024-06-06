import torch
from torch import nn
from einops import rearrange, repeat
import numpy as np
from torchinfo import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device =  'cpu'
from tool import load_config
#首先编写通道编码器,其主要作用就是将输入维度为[batch_size, channels, depths, height, width]的模型进行编码
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
        # print(q.shape)
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
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dims,
                 num_heads,
                 attn_dropout):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads =  num_heads
        self.attn_dropout = attn_dropout
        self.layernorm = nn.LayerNorm(normalized_shape=self.embedding_dims).to(device)
        self.multiheadattention = multiheadattention(d_model=self.embedding_dims,
                                                     n_heads=self.num_heads,
                                                     mask=None)
    def forward(self,query, key,value):
        x =  self.multiheadattention(query, key,value)
        return query + self.layernorm(x)

class Tublet_Embedding(nn.Module):
    def __init__(self, t, h, w, patch_t, patch_w, patch_h, embedding_dims, in_channel, batch_size):
        super().__init__()
        self.nt = t // patch_t
        self.nh = h // patch_h
        self.nw = w // patch_w
        self.inner_dims = in_channel*patch_h*patch_t*patch_w
        self.cov_layer = nn.Conv3d(
            in_channels=3,
            out_channels=self.inner_dims,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),dtype=torch.float32,
        ).to(device)
        #经过卷积之后，维度变为[batch_size, embedding_dims, nt, nw, nh]
        self.faltten = nn.Flatten(start_dim=3, end_dim=4).to(device)
         #经过flatten之后，维度变为[batch_size, inner_dims, nt, nw*nh] -> 做一个维度变换 [batch_size, nt, nw*nh, inner_dim]
        self.linear = nn.Linear(in_features=self.inner_dims, out_features=embedding_dims).to(device)
        self.class_embeddings = nn.Parameter(torch.randn(size=(1,1, self.nh*self.nw,self.inner_dims))).repeat(int(batch_size),1,1,1).to(device)
        self.position_embedding = nn.Parameter(torch.randn(size=(int(batch_size), self.nt + 1, self.nh*self.nw, self.inner_dims))).to(device)
    def forward(self, x): 
        x = self.faltten(self.cov_layer(x))
        # print(f'after cov and faltten {x.shape}')
        x = rearrange(x, 'b d t s -> b t s d')
        # print(x.shape, self.class_embeddings.shape,self.position_embedding.shape)
        x = torch.concat((self.class_embeddings, x),dim=1) + self.position_embedding
        x = self.linear(x)
        return rearrange(x, 'b t s d-> b (t s) d')
    

class Transformer(nn.Module):
    def __init__(self, atten_dropout, num_heads, mlp_dim, mlp_dropout, embedding_dims):
        super(Transformer,self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.mutiheadattention = MultiHeadSelfAttentionBlock(embedding_dims=embedding_dims,
                                                       num_heads=num_heads,
                                                       attn_dropout=atten_dropout).to(device)
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dims).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dims,out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_dim,out_features=embedding_dims),
            nn.Dropout(p=mlp_dropout)
        ).to(device)
    def forward(self, x):
        #这里的输入是编码完成的toknes，其维度为[batch_size, nt*nh*nw, embedding_dims]
        input = x
        x = self.mutiheadattention(x,x,x)
        x = self.layernorm(x + input)
        x = self.layernorm(x + self.mlp(x))
        return x

#垮视角注意力编码,
class CrossViewAttentionEncoder(nn.Module):
    def __init__(self, mlp_dim,
                  mlp_dropout, 
                  atten_dropout, 
                  fusion_layer:list,
                  embedding_dim:list,
                  num_heads:list,
                  num_layer:list):
        """该部分的输入应该是普通注意力机制的输入.
        在经过管道编码之后，各个视角的输入维度变为[batch_size, nt, nh*nw, embedding_dims],
        在输入之前需要全部转换为[batch_size, nt*nh*nw, embedding_dims]"""
        super().__init__()
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.fusion_layer = fusion_layer
        self.atten_dropot = atten_dropout
        self.embedding_dim_list = embedding_dim
        self.num_heads_list = num_heads
        self.num_layer_list = num_layer
        self.mha_self_attention = nn.ModuleList([MultiHeadSelfAttentionBlock(embedding_dims=self.embedding_dim_list[view_idx],
                                           num_heads=self.num_heads_list[view_idx],
                                           attn_dropout=self.atten_dropot) for view_idx in range(3)])
    def do_self_attention(self, tokens, cur_layer):
        """tokens的维度为[views, batch_size, nt*nh*nw, embedding_dims],是一个列表"""
        for view_idx, token in enumerate(tokens):
            if cur_layer >= self.num_layer_list[view_idx]:
                #如果当前层小于需要融合的层的话，正常做注意力机制就可以
                continue
            else:
                x = nn.LayerNorm(normalized_shape=self.embedding_dim_list[view_idx]).to(device)(token)
                x= self.mha_self_attention[view_idx].to(device)(query=x, key=x, value=x)
                x = nn.Dropout(p = self.atten_dropot)(x)
                #残差链接
                tokens[view_idx] = tokens[view_idx] + x
        return tokens
    def do_cross_attention(self, tokens,count):
        #融合的时候统一从后往前融合，view1, view2, view3的排列是按照tokens的从少到多（视角的从大到小）
        #tokens是来自后一个view(i+1)个tokens
        view_indices = [2, 1]#融合是降序的
        #kv是由较小的视角提供，q是由较大的视角提供
        view_idx = view_indices[count]
        query_view_index = view_idx - 1
        key_value_view_index = view_idx
        query = tokens[query_view_index]
        key, value = tokens[key_value_view_index], tokens[key_value_view_index]
        # print(f'query.shape = {query.shape}, key.shape = {key.shape}, value.shape = {value.shape}')
            #将kv投影到q的维度
        key = nn.Linear(in_features=key.shape[-1], out_features=query.shape[-1]).to(device)(key)
        value = nn.Linear(in_features=value.shape[-1], out_features=query.shape[-1]).to(device)(value)
            #每个view的编码维度，多头数量都是不一样的
        query = rearrange(nn.Linear(in_features=query.shape[1], out_features=key.shape[1]).to(device)(rearrange(query, 'b t d -> b d t')), 'b d t -> b t d')
        y = MultiHeadSelfAttentionBlock(
                embedding_dims=self.embedding_dim_list[query_view_index],
                num_heads=self.num_heads_list[query_view_index],
                attn_dropout=self.atten_dropot).to(device)(query=query,key=key,value=value)
        tokens[query_view_index] = rearrange(nn.Linear(in_features=tokens[query_view_index].shape[1], out_features=y.shape[1]).to(device)(rearrange(tokens[query_view_index], 'b t d -> b d t')), 'b d t -> b t d')
        return tokens
    def FeedForward(self, tokens, cur_layer):
        #这个是注意力机制或者交叉注意力机制的输入进行融合,其输入为[batch_size, nt*nh*nw, embedding_dims]
        for view_idx, x in enumerate(tokens):
            if cur_layer >= self.num_layer_list[view_idx]:
                continue
            else:
                y = nn.LayerNorm(normalized_shape=x.shape[-1]).to(device)(x)
                y = nn.Dropout(p=self.mlp_dropout)(y)
                y = nn.Linear(in_features=y.shape[-1], out_features=y.shape[-1]).to(device)(y)
                tokens[view_idx] = tokens[view_idx] + y
        return tokens
    def forward(self, tokens, cur_layer,count):
        """实例化CrossViewEncoder,这里的输入的token是经过管道编码之后的，其维度为[view, batch_size, nt*nw*nh, embedding_dims]"""
        # print(f'tokens.shape = {tokens[0].shape},{tokens[1].shape},{tokens[2].shape}')
        tokens = self.do_cross_attention(tokens=tokens,count=count)
        tokens = self.do_self_attention(tokens=tokens, cur_layer=cur_layer)
        tokens = self.FeedForward(tokens=tokens, cur_layer=cur_layer)
        return tokens

class MultiviewEncoder(nn.Module):
    def __init__(self, fusion_layer:list,
                 mlp_dim, mlp_dropout,
                 atten_dropout,
                 embedding_dims:list,
                 num_heads:list,
                 num_layer:list,
                 view_number:list):
        super().__init__()
        self.fusion_layer_list = fusion_layer
        self.num_layer_list = num_layer
        self.max_num_layer = max(self.num_layer_list)
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.atten_dropout = atten_dropout
        self.embedding_dim_list = embedding_dims
        self.num_heads_list = num_heads
        self.view_number = view_number
        self.crossviewattention = CrossViewAttentionEncoder(mlp_dim=self.mlp_dim,
                                               mlp_dropout=self.mlp_dropout,
                                               atten_dropout=self.atten_dropout,
                                               fusion_layer=self.fusion_layer_list,
                                               embedding_dim=self.embedding_dim_list,
                                               num_heads=self.num_heads_list,
                                               num_layer=self.num_layer_list)
        self.tramformer = nn.ModuleList([Transformer(atten_dropout=self.atten_dropout,
                                                       num_heads=self.num_heads_list[view_idx],
                                                       mlp_dim=self.mlp_dim,
                                                       mlp_dropout=self.mlp_dropout,
                                                       embedding_dims=self.embedding_dim_list[view_idx]) for view_idx in range(3)] )
        self.tub_embed = nn.ModuleList([Tublet_Embedding(t=8,h=112,w=112,
                                               patch_h=16,patch_t=self.view_number[view_idx],patch_w=16,
                                               embedding_dims=self.embedding_dim_list[view_idx],
                                               in_channel=3,
                                               batch_size=16) for view_idx in range(3)])
    def build_with_cross_view_attention(self, tokens):
        """将需要融合的各个视角层进行融合,其输入维度为[view_num, batch_size, nt*nw*nh, embeddings],注意，这里的第二个维度
        和第三个维度的值会根据视角的不同而改变"""
        for lyr in range(self.max_num_layer):
            print('第几层')
            count = 0
            if lyr in self.fusion_layer_list:
                print(f'当前融合层为{lyr}')
                # print(f'当前tokens的维度为{tokens[0].shape},{tokens[1].shape},{tokens[2].shape}')
                tokens = self.crossviewattention(tokens=tokens, cur_layer=lyr, count=count)
                count += 1
            else:
                for view_idx in range(len(tokens)):
                    if lyr >= self.num_layer_list[view_idx]:
                        continue
                    else:
                        tokens[view_idx] = self.tramformer[view_idx](tokens[view_idx])
        return tokens
    def forward(self,tokens):
        #这里的输入是最原始的数据，即[views_num, batchi_size, in_channel, depths, height, width]
        #首先经过管道编码，数据变为[view_nums, batch_size, nt*nw*nh, embedding_dims]
        for view_idx, token in enumerate(tokens):
            # print(f'token.shape = {token.shape}')
            tokens[view_idx]=self.tub_embed[view_idx](token)
        #返回的维度为[View_num, batch_size, nt*nw*nh, embedding_dims]
        for each_view_point in tokens:
            print(f'第几次')
            # print(f'每个视角的维度:{each_view_point.shape}')
            assert len(tokens) == 3, "tokens应该只有三个视角"
            assert each_view_point.ndim == 3, "每一个视角的输入维度应该是3"
        return self.build_with_cross_view_attention(tokens=tokens)

class Global_encoder(nn.Module):
    def __init__(self,paras_for_global_encoder,embedding_dims:list):
        super(Global_encoder,self).__init__()
        self.embedding_dim_for_global_encoder=paras_for_global_encoder['embedding_dims']
        self.mlp_dim_for_global_encoder=paras_for_global_encoder['mlp_dims']
        self.mlp_dropout_for_global_encoder=paras_for_global_encoder['mlp_dropout']
        self.num_head_for_global_encoder=paras_for_global_encoder['num_heads']
        self.atten_dropout_for_global_encoder=paras_for_global_encoder['atten_dropout']
        self.embedding_dims = embedding_dims
        self.normal_transformer = Transformer(
            atten_dropout=self.atten_dropout_for_global_encoder,
            num_heads=self.num_head_for_global_encoder,
            mlp_dim=self.mlp_dim_for_global_encoder,
            mlp_dropout=self.mlp_dropout_for_global_encoder,
            embedding_dims=self.embedding_dim_for_global_encoder)
        #维度转化
        self.ff_mve = nn.Linear(in_features=1024, out_features=1024).to(device)
        self.ff_view = nn.ModuleList([nn.Linear(in_features=self.embedding_dims[view_idx], out_features=self.embedding_dim_for_global_encoder).to(device) for view_idx in range(3)])
    def forward(self, x):
        #转换维度
        for view_idx, token in enumerate(x):
            x[view_idx] = self.ff_view[view_idx](token)
        token = torch.concat(x, dim=1)
        # torch.Size([8, 6528, 3072])
        token = self.ff_mve(token) #torch.Size([8, 6528, 1024])
        token = self.normal_transformer(token)
        return token
class MultiViewModel(nn.Module):
    def __init__(self,fusion_layer:list,
                 mlp_dims:list,
                 mlp_dropout,
                 num_heads:list,
                 num_layer:list,
                 atten_dropout,
                 embedding_dims:list,
                 paras_for_global_encoder:dict,
                 view_number:list):
        super().__init__()
        self.fusion_layer_list = fusion_layer
        self.mlp_dims = mlp_dims
        self.attn_dropout = atten_dropout
        self.mlp_dropout = mlp_dropout
        self.num_heads_list = num_heads
        self.num_layer_list = num_layer
        self.embedding_dims_list = embedding_dims
        self.paras_for_global_encoder = paras_for_global_encoder
        self.view_number = view_number

        self.multimviewencoder = MultiviewEncoder(fusion_layer=self.fusion_layer_list,
                                  mlp_dim=self.mlp_dims,
                                  atten_dropout=self.attn_dropout,
                                  mlp_dropout=self.mlp_dropout,
                                  embedding_dims=self.embedding_dims_list,
                                  num_heads=self.num_heads_list,
                                  num_layer=self.num_layer_list,
                                  view_number=self.view_number)
        
        self.global_encoder = Global_encoder(paras_for_global_encoder=self.paras_for_global_encoder,embedding_dims=self.embedding_dims_list)
        self.ff1 = nn.Linear(in_features=1024, out_features=12)
        self.ff2 = nn.Linear(in_features=12, out_features=12)
    def forward(self, x):
        #输入这里的维度为[view_num, batch_size, in_channel, depths, height, weight]
        token = self.multimviewencoder(x)
        token = self.global_encoder(token)
        token =self.ff1.to(device)(token)
        #token.shape = torch.Size([8, 441,12])
        # token = token[:,0,:]
        # token = self.ff2.to(device)(token)
        # token = rearrange(token, ' b d t-> b t d')
        return token
    
