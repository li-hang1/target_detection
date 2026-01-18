import torch
import torch.nn.functional as F

from utils.assign_anchors import assign_anchors

def rpn_loss(rpn_cls_logits, rpn_bbox_pred, anchors, gt_boxes, positive_iou=0.7, negative_iou=0.3, lambda_reg=1.0, lambda_cls_pos=1.0, lambda_cls_neg=1.0,):
    """
    rpn_cls_logits: tensor, shape [B, 2 * k, H/stride_H, W/stride_W]
    rpn_bbox_pred: tensor,shape [B, 4 * k, H/stride_H, W/stride_W]
    anchors: tensor, shape [num_anchors, 4],  (x1, y1, x2, y2)
    gt_boxes: List[Tensor[num_gt, 4]],  (x1, y1, x2, y2)
    positive_iou: float, Anchors with an IoU ≥ positive_iou are labeled as positive samples.
    negative_iou: float, Anchors with an IoU < negative_iou are labeled as negative samples.
    lambda_reg: float, The weight of the regression loss
    lambda_cls_pos: float, The weight of positive samples in the loss function.
    lambda_cls_neg: float, The weight of negative samples in the loss function.
    return:
        tensor, shape (1, ), The average RPN loss of samples within a batch.
    """
    B = rpn_cls_logits.shape[0]
    device = rpn_cls_logits.device

    all_cls_loss, all_reg_loss = 0., 0.

    for i in range(B):
        labels, bbox_targets = assign_anchors(anchors, gt_boxes[i], positive_iou, negative_iou)

        cls_logits = rpn_cls_logits[i].permute(1, 2, 0).reshape(-1, 2)

        pos_mask = labels == 1
        neg_mask = labels == 0

        cls_loss = torch.tensor(0.0, device=device)

        if pos_mask.any():
            pos_loss = F.cross_entropy(cls_logits[pos_mask], labels[pos_mask], reduction="mean")
            cls_loss += lambda_cls_pos * pos_loss

        if neg_mask.any():
            neg_loss = F.cross_entropy(cls_logits[neg_mask], labels[neg_mask], reduction="mean")
            cls_loss += lambda_cls_neg * neg_loss

        if pos_mask.any():
            bbox_pred = rpn_bbox_pred[i].permute(1, 2, 0).reshape(-1, 4)
            reg_loss = F.smooth_l1_loss(bbox_pred[pos_mask], bbox_targets[pos_mask], reduction="mean")
        else:
            reg_loss = torch.tensor(0.0, device=device)

        all_cls_loss += cls_loss
        all_reg_loss += reg_loss

    return (all_cls_loss / B) + lambda_reg * (all_reg_loss / B)


if __name__ == '__main__':
    rpn_cls_logits = torch.randn(2, 4, 2, 2)
    rpn_bbox_pred = torch.randn(2, 8, 2, 2)
    anchors = torch.randn(8, 4)
    gt_boxes = [torch.randn(1, 4), torch.randn(1, 4)]
    loss = rpn_loss(rpn_cls_logits, rpn_bbox_pred, anchors, gt_boxes)
    print(loss)

