import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from Conv2model import Conv2Model
from read import ReadHandler
from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #使用cuda
num_epoch=20
num_classes=68,
batch_size=20
learning_rate=0.001

file_list=['05','07','09','27','29']
train_feature=[0]*5
train_target=[0]*5
for i in range(5):
    readfile=ReadHandler('PIE dataset/Pose{id}_64x64.mat'.format(id=file_list[i]))
    train_feature[i],train_target[i]=readfile.read_train()
    train_feature[i]=torch.Tensor(train_feature[i]).view(-1,1,64,64).float().to(device)
    train_target[i]=torch.Tensor(train_target[i]).long().to(device)
    print(len(train_target[i]))

read_feature=torch.cat((train_feature[0],train_feature[1],train_feature[2],train_feature[3],train_feature[4]),0)
read_target=torch.cat((train_target[0],train_target[1],train_target[2],train_target[3],train_target[4]),0)
train_feature=read_feature
train_target=read_target
print(train_feature.shape,train_target.shape)


train_dataset= TensorDataset(train_feature,train_target)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


model=Conv2Model().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

total=len(train_target)

for epoch in range(num_epoch):
    total_correct=0
    for step,(feature,target) in enumerate(train_loader):
        feature=Variable(feature)
        target=Variable(target)

        output=model(feature)
        loss=criterion(output,target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size=target.size(0)
        _,predict=torch.max(output,1)
        total_correct+=(predict==target).sum().item()
    print('epoch:{} accuracy:{}'.format(epoch+1,total_correct/total))
torch.save(model.state_dict(),'gyc_all_conv2d.pt')
