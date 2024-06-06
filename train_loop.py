#模型训练和测试
from tqdm.auto import tqdm
# from torchvision.transforms import Resize, Compose, ToTensor
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
import torch
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist
def train(model:torch.nn.Module,
        train_dataloader:DataLoader,
          Epoches:int,
          device):
    ddp_loss = 0
    ddp_acc = 0
    print("Start training!!")
    # train_dataloader.sampler.set_e
    optimizer = torch.optim.SGD(params=model.parameters(),lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(Epoches):
        print(f'第{i}轮训练开始...')
        model.train()
        for item, data in tqdm(enumerate(train_dataloader)):
            X_rgb, y = data
            torch.cuda.empty_cache()
            X_rgb,  y = X_rgb.to(torch.float), y.to(torch.float)
            X_rgb, y = rearrange(X_rgb,'b t h w c -> b c t h w').to(device), y.to(device)
            optimizer.zero_grad()
            y_pred_logit = model(X_rgb)
            loss = loss_fn(y_pred_logit,y)
            loss.backward()
            optimizer.step()
            acc = (y_pred_logit.softmax(dim=1).argmax(dim=1) == y.argmax(1)).sum() / (y.shape[0])
            ddp_loss += loss.item()
            ddp_acc += acc
            if (item+1) % int(len(train_dataloader)) == 0:
                 # 用all_reduce收集所有进程上的某个参数的值，op表示收集操作，这里使用SUM来求所有loss的和
                step_loss = ddp_loss / int(len(train_dataloader))
                step_acc = ddp_acc / int(len(train_dataloader))
                with open ('E:/mdx/SelfAttention_ActionRecognition/Result/ResNet_4layer_De16_lr0001.txt','a+') as f:
                    f.write(f'{step_loss} {step_acc}\n')
                    f.close()
                print(f'step:{item}, loss:{step_loss}, acc:{step_acc}')
                ddp_loss = 0
                ddp_acc = 0
                # print(f'step:{item}, loss:{step_loss}, acc:{step_acc}')

    return step_loss, step_acc