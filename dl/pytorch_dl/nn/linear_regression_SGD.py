import torch as torch
import numpy as np
import torch.autograd.function as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# plt.plot(x_train, y_train, 'bo')
plt.scatter(x_train, y_train)
plt.show()

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

x_train = Variable(x_train)
y_train = Variable(y_train)

# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True)  # 随机初始化
b = Variable(torch.zeros(1), requires_grad=True)  # 使用 0 进行初始化


def linear_model(x):
    return x * w + b


y_ = linear_model(x_train)

plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()


def get_loss(y1, y2):
    return torch.mean((y1 - y2) ** 2)


loss = get_loss(y_, y_train)

print("loss = ", loss)

loss.backward()
print(w.grad, b.grad)

w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data

y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()


for e in range(10):
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)

    w.grad.zero_()
    b.grad.zero_()
    loss.backward()

    w.data = w.data - 1e-2 * w.grad.data
    b.data = b.data - 1e-2 * b.grad.data
    print('epoch: {}, loss: {}'.format(e, loss))

y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()

def linear_regression():
    pass


if __name__ == '__main__':
    linear_regression()
