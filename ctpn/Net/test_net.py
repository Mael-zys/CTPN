import sys
sys.path.append("/home/zhangyangsong/OCR/CTPN-master/ctpn/Net")
import torch.nn as nn
import torch.nn.functional as F
import img2col
import numpy as np
import torch
from torch.autograd import Variable
from roi_align.Roi_align import RoIAlign
import math
def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

class VGG_16(nn.Module):
    """
    VGG-16 without pooling layer before fc layer
    VGG16进行底层特征提取
    """
    def __init__(self):
        super(VGG_16, self).__init__()
        # CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, \
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.convolution1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.convolution1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.convolution2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convolution2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.convolution3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convolution3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convolution3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.convolution4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convolution4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.convolution5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convolution5_3 = nn.Conv2d(512, 512, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convolution1_1(x), inplace=True)
        x = F.relu(self.convolution1_2(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2_1(x), inplace=True)
        x = F.relu(self.convolution2_2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3_1(x), inplace=True)
        x = F.relu(self.convolution3_2(x), inplace=True)
        x = F.relu(self.convolution3_3(x), inplace=True)
        x = self.pooling3(x)
        x = F.relu(self.convolution4_1(x), inplace=True)
        x = F.relu(self.convolution4_2(x), inplace=True)
        x = F.relu(self.convolution4_3(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution5_1(x), inplace=True)
        x = F.relu(self.convolution5_2(x), inplace=True)
        x = F.relu(self.convolution5_3(x), inplace=True)
        return x


class BLSTM(nn.Module):
    """
    双向LSTM，增强关联序列的信息学习
    """
    def __init__(self, channel, hidden_unit, bidirectional=True):
        """
        :param channel: lstm input channel num
        :param hidden_unit: lstm hidden unit
        :param bidirectional:
        """
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        """
        WARNING: The batch size of x must be 1.
        """
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('VGG_16', VGG_16())
        self.rnn = nn.Sequential()
        self.rnn.add_module('im2col', img2col.Im2col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BLSTM(3 * 3 * 512, 128))
        self.FC = nn.Conv2d(256, 50, 1)
        self.vertical_coordinate = nn.Conv2d(500, 2*10, 1)  # 最终输出2K个参数（k=10），10表示anchor的尺寸个数，2个参数分别表示anchor的h和dy
        self.score = nn.Conv2d(500, 2*10 , 1)  # 最终输出是2K个分数（k=10），2表示有无字符，10表示anchor的尺寸个数
        self.side_refinement = nn.Conv2d(500, 1*10, 1)  # 最终输出1K个参数（k=10），该参数表示该anchor的水平偏移，用于精修文本框水平边缘精度，10表示anchor的尺寸个数
        self.dd = nn.Linear(7,1)
    def forward(self, x, boxes_data, box_index_data ,val=False):
        """
        前向传播：图像——>CNN——>RNN——>FC——>返回vertical_pred、score、side_refinement
        """
        # h=math.ceil(x.shape[2]/16)
        # w=math.ceil(x.shape[3]/16)
        x = self.cnn(x)
        h = x.shape[2]
        w = x.shape[3]
        x = self.rnn(x)
        x = self.FC(x)
        x = F.relu(x, inplace=True)
        # x=x.repeat(10,1,1,1)
        # x = Variable(x, requires_grad=True)
        # is_cuda=True
        # boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
        # box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)
        # a=RoIAlign(7, 1, transform_fpcoor=True)
        # x=a(x, boxes, box_index)
        # x=x.reshape((-1,7))
        # x=self.dd(x)

        # x=x.reshape(10,512,h,w)

        x=x.repeat(10,1,1,1)
        x = Variable(x, requires_grad=True)
        is_cuda=True

        boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
        box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)
        a=RoIAlign(7, 1, transform_fpcoor=True)
        x=a(x, boxes, box_index)
        x=x.reshape((-1,7))
        x=self.dd(x)

        x=x.reshape(10,h,w,50)
        # x=x.transpose(0,3,1,2)
        x=x.transpose(2,3)
        x=x.transpose(1,2)
        x=x.reshape(1,500,h,w)


        vertical_pred = self.vertical_coordinate(x)  # 垂直坐标预测
        # vertical_pred=vertical_pred.reshape(1,20,h,w)
        score = self.score(x)  # 得分score
        # score = score.reshape(1,20,h,w)
        if val:  # 这里是做什么处理呢？  在infer的时候用到
            score = score.reshape((score.shape[0], 10, 2, score.shape[2], score.shape[3]))
            score = score.squeeze(0)
            score = score.transpose(1, 2)
            score = score.transpose(2, 3)
            score = score.reshape((-1, 2))
            #score = F.softmax(score, dim=1)
            score = score.reshape((10, vertical_pred.shape[2], -1, 2))
            vertical_pred = vertical_pred.reshape((vertical_pred.shape[0], 10, 2, vertical_pred.shape[2],
                                                   vertical_pred.shape[3]))
            vertical_pred = vertical_pred.squeeze(0)
        side_refinement = self.side_refinement(x)  # 边缘优化
        # side_refinement = side_refinement.reshape(1,10,h,w)
        return vertical_pred, score, side_refinement
