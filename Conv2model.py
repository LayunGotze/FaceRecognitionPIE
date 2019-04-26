from read import ReadHandler
import torch 
import torchvision.models as models
import torch.nn as nn

class Conv2Model(nn.Module):
    def __init__(self):
        super(Conv2Model,self).__init__()
        self.model1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=6,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6,out_channels=18,kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.fc1=nn.Linear(8*8*18,500)
        self.fc2=nn.Linear(500,68)
        self.fc=nn.Sequential(
            nn.Linear(8*8*18,500),
            nn.Dropout(0.05),
            nn.Linear(500,68)
        )
    def forward(self,x):
        x=self.model1(x)
        x=x.view(-1,8*8*18)
        x=self.fc(x)
        return x

#convmodel=Conv1Model()
"""
read=ReadHandler('PIE dataset/Pose29_64x64.mat')
input_fea,input_label=read.read_train()
input=torch.Tensor(input_fea).view(-1,64,64)
"""
"""
input=torch.ones(10,1,64,64)
model=Conv2Model()
print(model(input).shape)
"""
