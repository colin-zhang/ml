import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

w_target = np.array([0.5, 3, 2.4])  # 定义参数
b_target = np.array([0.9])  # 定义参数

# y = w0*x + w1*x^2 + w2*x^3 + b


x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3

print(x_sample, x_sample.shape)

plt.plot(x_sample, y_sample, label='real curve')
plt.legend()
plt.show()

# 构建数据 x 和 y
# x 是一个如下矩阵 [x, x^2, x^3]
# y 是函数的结果 [y]

x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
print(x_train)
x_train = torch.from_numpy(x_train).float()  # 转换成 float tensor
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)  # 转化成 float tensor

# 定义参数和模型
w = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

# 将 x 和 y 转换成 Variable
x_train = Variable(x_train)
y_train = Variable(y_train)


def multi_linear(xx, ww, bb):
    return torch.mm(xx, ww) + bb


def get_loss(y1, y2):
    return torch.mean((y1 - y2) ** 2)


# 画出更新之前的模型
y_pred = multi_linear(x_train, w, b)

plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()
plt.show()

loss = get_loss(y_pred, y_train)
loss.backward()
print(type(loss), loss.dim(), loss.item())

for e in range(10000):
    y_pred = multi_linear(x_train, w, b)
    loss = get_loss(y_pred, y_train)

    w.grad.zero_()
    b.grad.zero_()
    loss.backward()

    w.data = w.data - 1e-3 * w.grad.data
    b.data = b.data - 1e-3 * b.grad.data

y_pred = multi_linear(x_train, w, b)
plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()
plt.show()
