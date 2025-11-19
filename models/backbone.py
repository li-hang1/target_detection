from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.body = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.out_channels = 2048
    def forward(self, x):
        return self.body(x)