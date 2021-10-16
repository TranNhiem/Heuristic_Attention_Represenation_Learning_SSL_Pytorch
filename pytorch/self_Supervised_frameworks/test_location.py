import torch
from torch.nn import functional as F

"""手动定义卷积核(weight)和偏置"""
w = torch.rand(1, 4, 3, 3)  # 16种3通道的5乘5卷积核

b = torch.ones(1)  # 和卷积核种类数保持一致(不同通道共用一个bias)

"""定义输入样本"""
x = torch.tensor([[[[1, 2, 3, 3, 1], [3, 2, 1, 2, 2], [1, 2, 3, 1, 1], [3, 2, 1, 2, 2], [3, 2, 1, 2, 2]]]],
                  # [[1, 2, 3, 3, 1], [3, 2, 1, 2, 2], [1, 2, 3, 1, 1], [3, 2, 1, 2, 2], [3, 2, 1, 2, 2]],
                  # [[1, 2, 3, 3, 1], [3, 2, 1, 2, 2], [1, 2, 3, 1, 1], [3, 2, 1, 2, 2], [3, 2, 1, 2, 2]]]],
                 dtype=torch.float32)  # 1张3通道的28乘28的图像

image = torch.rand(1, 3, 5, 5)
print(x.shape)
print(image.shape)

inputs = torch.cat([x ,image] ,axis=1)

# x = torch.randn(1, 3, 5, 5)
print(inputs)
"""2D卷积得到输出"""
# out = F.conv2d(x, w, b, stride=1, padding=1)  # 步长为1,外加1圈padding,即上下左右各补了1圈的0,
# print(out)
#
# out = F.conv2d(x, w, b, stride=2, padding=2)  # 步长为2,外加2圈padding
# print(out)
out = F.conv2d(inputs, w)  # 步长为1,默认不padding, 不够的舍弃，所以对于28*28的图片来说，算完之后变成了24*24
print(out)
