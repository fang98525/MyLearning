"""集成学习解决过拟合开销大  采用dropout 随机剔除一些神经元，不参与训练"""

import torch
#variable是 pytorch里面存放会变化的值的池子 ，刚好符合权重的动态更新   tensor不能反向传播，variable可以反向传播
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

lr = 0.1

#20 个样本
N_SAMPLES = 20
N_HIDDEN = 300

# 训练数据
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
x, y = Variable(x), Variable(y)

# 测试数据
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
test_x, test_y = Variable(test_x, volatile=True), Variable(test_y, volatile=True)

# 展示一下数据分布
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train set')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test set')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop out 0.5   每一层都可以dropout一下
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop out 0.5
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

print(net_overfitting)  # 会过拟合的网络结构
"""
Sequential (
  (0): Linear (1 -> 300)
  (1): ReLU ()
  (2): Linear (300 -> 300)
  (3): ReLU ()
  (4): Linear (300 -> 1)
)
"""


print(net_dropped)      # 使用了Dropout的网络结构
"""
Sequential (
  (0): Linear (1 -> 300)
  (1): Dropout (p = 0.5)
  (2): ReLU ()
  (3): Linear (300 -> 300)
  (4): Dropout (p = 0.5)
  (5): ReLU ()
  (6): Linear (300 -> 1)
)
"""


#定义了两个不同的网络   一个使用了dropout进行泛化   另一个没有会过拟合
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=lr)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=lr)
loss_func = torch.nn.MSELoss()

plt.ion()   # hold柱图

for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        # 切换到测试形态
        net_overfitting.eval()
        net_dropped.eval()

        # 画一下
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train set')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test set')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='no dropout')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        # plt.text(0, -1.2, 'no dropout loss=%.4f' % loss_func(test_pred_ofit, test_y).data[0], fontdict={'size': 20, 'color':  'red'})
        # plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data[0], fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5));plt.pause(0.1)

        # 切换回训练形态
        net_overfitting.train()
        net_dropped.train()

plt.ioff()
plt.show()
