from tqdm.auto import tqdm
# from torchvision.transforms import Resize, Compose, ToTensor
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
import torch
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(model:torch.nn.Module,
        train_dataloader:DataLoader,
          Epoches:int):
    ddp_loss = 0
    ddp_acc = 0
    print("Start training!!")
    model.to(device)
    # train_dataloader.sampler.set_e
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.00001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(Epoches):
        print(f'第{i}轮训练开始...')
        model.train()
        for item, (X_Skeleton, y) in tqdm(enumerate(train_dataloader)):
            torch.cuda.empty_cache()
            X_Skeleton, y = X_Skeleton.to(torch.float).to(device), y.to(torch.float).to(device)
            optimizer.zero_grad()
            y_pred_logit = model(X_Skeleton)
            # print(f'y_pred_logit:{y_pred_logit.shape}, y:{y.shape}')
            loss = loss_fn(y_pred_logit,y)
            loss.backward()
            optimizer.step()
            acc = (y_pred_logit.softmax(dim=1).argmax(dim=1) == y.argmax(1)).sum() / (y.shape[0])
            del y_pred_logit
            ddp_loss += loss.item()
            ddp_acc += acc
            if (item+1) % 49 ==0:
                step_loss = ddp_loss /(49)
                step_acc = ddp_acc /(49) 
                with open('E:/mdx/SelfAttention_ActionRecognition/Result/Skeleton_B16_D4_lr00001.txt', 'a+') as f:
                    f.write(f'{step_loss} {step_acc}\n')
                    f.close()
                print(f'step:{item}, loss:{step_loss}, acc:{step_acc}')
                ddp_loss = 0
                ddp_acc = 0             
    return step_loss, step_acc