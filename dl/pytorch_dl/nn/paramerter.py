import numpy as np
import torch
from torch import nn

'''
一种非常流行的初始化方式叫 Xavier，方法来源于 2010 年的一篇论文:
Understanding the difficulty of training deep feedforward neural networks
其通过数学的推到，证明了这种初始化方式可以使得每一层的输出方差是尽可能相等的，有兴趣的同学可以去看看论文
'''

# 定义一个 Sequential 模型
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# 访问第一层的参数
w1 = net1[0].weight
b1 = net1[0].bias


class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )

        self.l1[0].weight.data = torch.randn(40, 30)  # 直接对某一层初始化

        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


net2 = sim_net()

# 访问 children
print("访问 children")
for i in net2.children():
    print(i)
# 访问 modules
print("访问 modules")
for i in net2.modules():
    print(i)


'''
torch.nn.init
因为 PyTorch 灵活的特性，我们可以直接对 Tensor 进行操作从而初始化，
PyTorch 也提供了初始化的函数帮助我们快速初始化，就是 torch.nn.init，
其操作层面仍然在 Tensor 上，
'''
from torch.nn import init
print(net1[0].weight)
init.xavier_uniform(net1[0].weight) # 这就是上面我们讲过的 Xavier 初始化方法，PyTorch 直接内置了其实现
print(net1[0].weight)
