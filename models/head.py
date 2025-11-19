from torch import nn
import torch.nn.functional as F

class FastRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * output_size[0] * output_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, num_classes + 4)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred
