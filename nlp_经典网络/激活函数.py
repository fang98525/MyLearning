import  torch
import matplotlib.pyplot as plt
import  torch.optim as optim
import torch.nn as nn

#
# #激活函数
# x=torch.range(-5,5,0.1)
# print(x)
# y=torch.sigmoid(x)
# y1=torch.tanh(x)
# y2=torch.relu(x)
# # y3=torch.softmax(x)
# # y1=torch.nn.Sigmoid(x)
# plt.plot(x.numpy(),y.numpy())
# # plt.plot(x.numpy(),y1.numpy())
# # plt.plot(x.numpy(),y2.numpy())
# # plt.plot(x.numpy(),y3.numpy())
#
# plt.show()



# adam优化器
class Perception(nn.Module):
    def __init__(self,input_dim):
        super(Perception,self).__init__()
        self.linear=nn.Linear(input_dim,1)
    def forwad(self,x):
        return  torch.sigmoid(self.linear(x)).squeeze()
input_num=2
lr=0.001
perception=Perception()
loss=nn.BCELoss()
optim=optim.Adam(perception.parameters(),lr=lr)
output=perception(input_num)
