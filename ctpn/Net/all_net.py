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
import torch.utils.model_zoo as model_zoo
import cv2
from torchvision import models
import torchvision

def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # 卷积参数变量初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # BN参数初始化
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        output={}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        output['x1']=x
        # print(x.shape)
        x = self.maxpool(x)
        
        
        x = self.layer1(x)
        # print(x.shape)
        output['x2']=x
        x = self.layer2(x)
        # print(x.shape)
        output['x3']=x
        x = self.layer3(x)
        # print(x.shape)
        output['x4']=x
        x = self.layer4(x)
        # print(x.shape)
        output['x5']=x

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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
        # self.res = nn.Sequential()
        # self.res.add_module('Resnet', resnet18(True))
        # self.cnn = nn.Sequential()
        # self.cnn.add_module('VGG_16', VGG_16())
        self.rnn = nn.Sequential()
        self.rnn.add_module('im2col', img2col.Im2col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BLSTM(3 * 3 * 1024, 128))
        self.FC = nn.Conv2d(256, 512, 1)
        self.vertical_coordinate = nn.Conv2d(512, 2 * 10, 1)  # 最终输出2K个参数（k=10），10表示anchor的尺寸个数，2个参数分别表示anchor的h和dy
        self.score = nn.Conv2d(512, 2 * 10, 1)  # 最终输出是2K个分数（k=10），2表示有无字符，10表示anchor的尺寸个数
        self.side_refinement = nn.Conv2d(512, 10, 1)  # 最终输出1K个参数（k=10），该参数表示该anchor的水平偏移，用于精修文本框水平边缘精度，10表示anchor的尺寸个数

    def forward(self, x, val=False):
        """
        前向传播：图像——>CNN——>RNN——>FC——>返回vertical_pred、score、side_refinement
        """
        # x = self.cnn(x)
        # x = self.res(x)

        # print(x.shape)
        x = self.rnn(x['x4'])
        print(x.shape)
        x = self.FC(x)
        x = F.relu(x, inplace=True)
        vertical_pred = self.vertical_coordinate(x)  # 垂直坐标预测
        score = self.score(x)  # 得分score
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
        return vertical_pred, score, side_refinement

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  
        x4 = output['x4']  
        x3 = output['x3']  

        score = self.relu(self.deconv1(x5))              
        score = self.bn1(score + x4)                      
        score = self.relu(self.deconv2(score))            
        score = self.bn2(score + x3)                      
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  

class second_roi(nn.Module):

    def __init__(self):
        super(second_roi, self).__init__()
        # CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, \
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.convolution1 = nn.Conv2d(512, 512, 1)
        self.convolution2 = nn.Conv2d(512, 512, 1)
        self.convolution3 = nn.Conv2d(512, 512, 1)
        self.note = nn.Linear(512*49,2)
        self.reg = nn.Linear(512*49,8)

    def forward(self, project):
        x=self.convolution1(project)
        x=self.convolution2(x)
        index=np.arange(project.shape[0])
        boxes=np.array([[0,0,13,13]])
        boxes=np.repeat(boxes,project.shape[0],axis=0)
        is_cuda=False
        boxes = to_varabile(boxes, requires_grad=False, is_cuda=is_cuda)
        box_index = to_varabile(index, requires_grad=False, is_cuda=is_cuda)
        a=RoIAlign(7, 7, transform_fpcoor=True)
        x=a(x, boxes, box_index)
        x=self.convolution3(x)
        x=x.reshape((-1,512*49))
        note=self.note(x)
        reg=self.reg(x)
        return note, reg

if __name__ == '__main__':
    res = resnet50(pretrained=True)
    ctpn = CTPN()
    a = np.random.rand(1,3,640,640)
    a = torch.FloatTensor(a)
    b = res(a)
    print(b['x4'].shape)
    c = ctpn(b)