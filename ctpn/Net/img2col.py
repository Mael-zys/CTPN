import torch.nn.functional as F
import torch.nn as nn


class Im2col(nn.Module):
    """
    将VGG最后一层卷积层输出的feature map转化为向量形式，用于接下来的LSTM训练
    """
    def __init__(self, kernel_size, stride, padding):
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        height = x.shape[2]
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x
