import numpy as np
import torch

# 导入 pytorch 内置的 mnist 数据
from torchvision.datasets import mnist

from torch import nn
from torch.autograd import Variable

train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

# a_data type is PIL.Image.Image
# torch.Tensor
a_data, a_label = train_set[0]
a_data.show()

a_data = np.array(a_data, dtype='float32')
print(a_data.shape)


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x


train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)  # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)


