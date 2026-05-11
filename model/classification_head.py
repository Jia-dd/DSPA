import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        # 全连接层
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
