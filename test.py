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

file_list=['05','07','09','27','29']
test_feature=[0]*5
test_target=[0]*5
for i in range(5):
    readfile=ReadHandler('PIE dataset/Pose{id}_64x64.mat'.format(id=file_list[i]))
    test_feature[i],test_target[i]=readfile.read_test()
    test_feature[i]=torch.Tensor(test_feature[i]).view(-1,1,64,64).float().to(device)
    test_target[i]=torch.Tensor(test_target[i]).long().to(device)
    print(len(test_target[i]))

read_feature=torch.cat((test_feature[0],test_feature[1],test_feature[2],test_feature[3],test_feature[4]),0)
read_target=torch.cat((test_target[0],test_target[1],test_target[2],test_target[3],test_target[4]),0)
test_feature=read_feature
test_target=read_target
print(test_feature.shape,test_target.shape)
total=len(test_target)

model=Conv2Model().to(device)
model.load_state_dict(torch.load('gyc_all_conv2d.pt'))
model.eval()

with torch.no_grad():
    output=model(test_feature)
    _,predict=torch.max(output,1)
    accuracy=(predict==test_target).sum().item()
    print(accuracy,total)
    print("accuracy:{}".format(accuracy/total))

#训练集上准确率0.9746638082245176
#测试集上准确率0.945046439628483