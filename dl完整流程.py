import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn

#样本数量
n=400

#生成测试用例
X=10*torch.rand([n,2])-5.0  #400*2 得二维 矩阵
print(X)
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y=X@(w0)+b0+torch.normal(0.0,2.0 ,size=[n,1])  # 生成x对应得400个目标值且带干扰

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)
ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()
# 构建数据管道
def data_iter(features,labels,batchsize=4):
    num_examples=len(features)
    indices=list(range(num_examples))
    np.random.shuffle(indices)#  实现随机读取样本
    for  i in range(0,num_examples,batchsize):#间距为batch大小
        indexs=torch.LongTensor(indices[i:min(i+batchsize,num_examples)])
        yield features.index_select(0,indexs)  ,labels.index_select(0,indexs)
        # index_select函数   0表示按行索引index进行查找张量    yield 表市每次返回一个了迭代得计算得数据
baychsize=7
(features,lables)=next(data_iter(X,Y,batchsize=baychsize)  ) #  使用next返回迭代器结果列表得第一个元素   若要全部输出使用for
print(features)
print(lables)
# 定义模型：
class myNet():
    def __init__(self):
        self.w=torch.rand_like(w0,requires_grad=True)
        self.b=torch.rand_like(b0,requires_grad=True)
    #正像传播foeward
    def forward(self,x):
        x= x@self.w + self.b
        return x

    #损失函数
    def loss_function(self,pred,true):
        return torch.mean((pred-true)**2/2)
model=myNet()

#训练模型
def train(model,features,lables):
    predictions=model.forward(features)
    loss=model.loss_function(predictions,lables)
    # 反向传播
    loss.backward()
    #使用torch.no_grad()避免梯度记录，也可以通过操作 model.w.data 实现避免梯度记录
    #梯度更新  相当于优化器
    with torch.no_grad():
        model.w-=0.001*model.w.grad
        model.b -= 0.001 * model.b.grad
    # 梯度清零   不然梯度会累加
        model.w.grad.zero_()
        model.b.grad.zero_()
    return loss  #这里值优化了一次


batch_size = 10
(features,labels) = next(data_iter(X,Y,batch_size))
loss=train(model,features,labels)
print(loss)

# 对损失训练epoch次   优化器是自动训练到最优参数
epoch=1000
for epoch in range(1,epoch+1):
    for features,lables in data_iter(X,Y,10):
        loss=train(model,features,lables)
    if epoch%100==0:
        print("第{}次训练得损失为：{}".format(epoch,loss))
        print("权重{}和偏值为{}".format(model.w.data,model.b.data))

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.plot(X[:,0].numpy(),(model.w[0].data*X[:,0]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)
ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.plot(X[:,1].numpy(),(model.w[1].data*X[:,1]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()