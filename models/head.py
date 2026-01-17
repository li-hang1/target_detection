import torch
from torch import nn
import torch.nn.functional as F

class FastRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * output_size[0] * output_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        """
        x: Tensor, shape [N_i, 2048, H_out, W_out], one of the outputs of roi_align
        return:
            cls_score: shape [N_i, num_classes + 1]
            bbox_pred: shape [N_i, num_classes * 4]
        """
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

if __name__ == '__main__':
    model = FastRCNNHead(2048, 6, (7, 7))
    x = torch.randn(2, 2048, 7, 7)
    cls_score, bbox_pred = model(x)
    print(f"cls_score shape: {cls_score.shape}, bbox_pred shape: {bbox_pred.shape}")
