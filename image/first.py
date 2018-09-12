import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('lena.jpg')  # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape  # (512, 512, 3)

plt.imshow(lena)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()

from PIL import Image

lena = mpimg.imread('lena.jpg')  # 这里读入的数据是 float32 型的，范围是0-1
im = Image.fromarray(np.uint8(lena * 255))
im.show()

#灰度图片
I = Image.open('lena.jpg')
I.show()
L = I.convert('L')
L.show()
