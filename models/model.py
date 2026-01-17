from torch import nn

from backbone import Backbone
from RPN import RPN
from head import FastRCNNHead
from utils.generate_proposals import generate_proposals
from utils.generate_anchors import generate_anchors
from utils.roi_align import roi_align


class FasterRCNN(nn.Module):
    def __init__(self, img_size, num_classes, output_size=(7, 7)):
        super().__init__()
        self.img_size = img_size  # tuple
        self.backbone = Backbone()
        self.output_size = output_size
        self.rpn = RPN(self.backbone.out_channels)
        self.head = FastRCNNHead(self.backbone.out_channels, num_classes, output_size)

    def forward(self, images):
        """
        images: tensor, shape [B, 3, H, W], original image
        """
        feature_map = self.backbone(images)
        H_f, W_f = feature_map.shape[2], feature_map.shape[3]
        stride_h, stride_w = self.img_size[0] / H_f, self.img_size[1] / W_f
        anchors = generate_anchors((H_f, W_f), (stride_h, stride_w), img_size=self.img_size, device=feature_map.device)
        rpn_cls, rpn_bbox = self.rpn(feature_map)
        proposals = generate_proposals(rpn_cls, rpn_bbox, anchors, self.img_size)
        rois = roi_align(feature_map, proposals, self.img_size, output_size=self.output_size)
        cls_score, bbox_pred = [], []
        for i in range(len(rois)):
            one_cls_score, one_bbox_pred = self.head(rois[i])
            cls_score.append(one_cls_score), bbox_pred.append(one_bbox_pred)
        return cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors