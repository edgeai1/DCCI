import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class EarlyNetBranchy1(nn.Module):
    def __init__(self, dropout=0.5):
        super(EarlyNetBranchy1, self).__init__()
        self.fc0 = nn.AdaptiveAvgPool2d(output_size=(208, 208))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=51, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        self.fc = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.fc1(x.view(batch_size, -1))
        return x


if __name__ == '__main__':
    model = EarlyNetBranchy1().to("cuda")
    example = (torch.randn(1, 51, 208, 208).to("cuda"))

    flops = FlopCountAnalysis(model, example)
    print("FLOPs: ", flops.total() / 1000 ** 3)
