import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from training.models.image_classification_model_base import ImageClassificationModelBase


class ResNet50Mse(ImageClassificationModelBase):
    """
        对于ResNet-50包装类的修改，用于mse损失的独热向量。
    """

    def __init__(self, out_size, pretrained=False):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)

        # 修改第一层卷积层的输入通道数使得接受灰度图
        # self.model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改最后一层全连接层的输出通道数
        self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=out_size))
        self.out_size = out_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.mse_loss(out, labels)  # Calculate loss
        # loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def accuracy(self, outputs, labels):
        correct = 0
        for output, label in zip(outputs, labels):
            composed_output = torch.split(output, 33) # 33是字符集的长度
            composed_label = torch.split(label, 33)

            prediction_correct = True
            for one_hot_out, one_hot_lbl in zip(composed_output, composed_label):
                one_hot_out = torch.unsqueeze(one_hot_out, 0)
                _, out_index = torch.max(one_hot_out, dim=1)
                one_hot_lbl = torch.unsqueeze(one_hot_lbl, 0)
                _, lbl_index = torch.max(one_hot_lbl, dim=1)

                if out_index != lbl_index:
                    prediction_correct = False
                    break
            if prediction_correct:
                correct += 1
        return torch.tensor(correct / len(outputs))

    # _, preds = torch.max(outputs, dim=1)
    # return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validation_step(self, batch):
        images, labels = batch
        out = self.model(images)  # Generate predictions
        loss = F.mse_loss(out, labels)  # Calculate loss
        # loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
