import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .image_classification_model_base import ImageClassificationModelBase


class ResNet50(ImageClassificationModelBase):
    """
        迁移学习
    """
    def __init__(self, out_size, pretrained=False):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)

        # 修改第一层卷积层的输入通道数使得接受灰度图
        #self.model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改最后一层全连接层的输出通道数
        self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=out_size))

    def forward(self, x):
        return self.model(x)
