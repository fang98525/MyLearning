import numpy as np
import matplotlib.pyplot as plt


#model  或者优化目标
def f(x1, x2):
    return x1**2+2*x1+x2**2+x2



#函数手动求导   即为梯度
def gradf(x1, x2):
    gradx1 = 2*x1+2
    gradx2 = 2*x2+1
    return gradx1,gradx2
#随机初始化参数值
x1, x2 = 1, 1
y = f(x1, x2) #5
y=8
print('初始y',y)

#学习率  eta
eta = 0.2

for step in range(1000):
    gradf1, gradf2 = gradf(x1, x2)
    x1t = x1 - eta*gradf1
    x2t = x2 - eta*gradf2
    yt = f(x1t, x2t)
    if (yt < y):  #终止条件   y=5为优化目标   权重更新    若y不在下降  则停止优化  earing stopping

        x1 = x1t
        x2 = x2t
        y = yt
    else:
        continue
    print("最终结果",step, x1, x2, y)


