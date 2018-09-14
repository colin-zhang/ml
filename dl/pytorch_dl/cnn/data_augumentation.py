from PIL import Image
from torchvision import transforms as tfs

# 读入一张图片
im = Image.open('./cat.png')
#im.show()
print(im)

# 比例缩放
print('before scale, shape: {}'.format(im.size))
new_im = tfs.Resize((100, 200))(im)
print('after scale, shape: {}'.format(new_im.size))
# new_im.show()

random_im1 = tfs.RandomCrop(100)(im)
#random_im1.show()

# 中心裁剪出 100 x 100 的区域
center_im = tfs.CenterCrop(100)(im)
# center_im.show()

# 随机水平翻转
h_filp = tfs.RandomHorizontalFlip()(im)
# h_filp.show()

# 随机竖直翻转
v_flip = tfs.RandomVerticalFlip()(im)
v_flip.show()

rot_im = tfs.RandomRotation(45)(im)
rot_im.show()

# 亮度
bright_im = tfs.ColorJitter(brightness=1)(im) # 随机从 0 ~ 2 之间亮度变化，1 表示原图
bright_im

# 对比度
contrast_im = tfs.ColorJitter(contrast=1)(im) # 随机从 0 ~ 2 之间对比度变化，1 表示原图
contrast_im

# 颜色
color_im = tfs.ColorJitter(hue=0.5)(im) # 随机从 -0.5 ~ 0.5 之间对颜色变化
color_im



im_aug = tfs.Compose([
    tfs.Resize(120),
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(96),
    tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
])

import matplotlib.pyplot as plt

nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(im_aug(im))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
plt.show()