import torch
from torch import nn


class FlattenLayer(nn.Module):

    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)


class PersonalModel(nn.Module):

    def __init__(self):

        super(PersonalModel, self).__init__()

        # 5×5의 커널을 사용해서 처음에 32개, 다음에 64개의 채널 작성
        # BatchNorm2d는 이미지용 Batch Normalization
        # Dropout2d는 이미지용 Dropout
        # 마지막으로 FlattenLayer 적용
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 5),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),
            FlattenLayer()
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200, 4)
        )

    def forward(self, x):
        out = self.conv_net(x)
        out = self.mlp(out)
        return out