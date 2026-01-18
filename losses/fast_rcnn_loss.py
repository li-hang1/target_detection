import torch
import torch.nn.functional as F

def fast_rcnn_loss(class_logits, bbox_pred, gt_labels, gt_boxes, num_classes, lambda_reg=1.0, lambda_cls_pos=1.0, lambda_cls_neg=1.0):
    """
    cls_score: List[Tensor[N_i, num_classes + 1]]
    bbox_pred: List[Tensor[N_i, num_classes * 4]]
    gt_labels: List[Tensor(N_i, )], The true classification labels corresponding to the proposals (0 = background, 1..K = foreground classes)
    gt_boxes: List[Tensor[N_i, 4]], (x1, y1, x2, y2)
    num_classes: int, The number of classes.
    lambda_reg: float, The weight of the regression loss
    lambda_cls_pos: float, The weight of positive samples in the loss function.
    lambda_cls_neg: float, The weight of negative samples in the loss function.
    return:
        tensor, shape (1, ), The average fast r cnn loss of samples within a batch.
    """
    all_cls_loss, all_reg_loss = [], []

    for logits, bbox_pred_i, labels, targets in zip(class_logits, bbox_pred, gt_labels, gt_boxes):

        pos_mask = labels > 0
        neg_mask = labels == 0

        cls_loss = torch.tensor(0.0, device=logits.device)

        if pos_mask.any():
            pos_loss = F.cross_entropy(logits[pos_mask], labels[pos_mask], reduction="mean")
            cls_loss += lambda_cls_pos * pos_loss

        if neg_mask.any():
            neg_loss = F.cross_entropy(logits[neg_mask], labels[neg_mask], reduction="mean")
            cls_loss += lambda_cls_neg * neg_loss

        pos_inds = torch.nonzero(pos_mask).squeeze(1)

        if pos_inds.numel() > 0:
            N_i, _ = bbox_pred_i.shape
            bbox_pred_i = bbox_pred_i.view(N_i, num_classes, 4)  # [N_i, num_classes, 4]

            cls_inds = labels[pos_inds] - 1                      # [num_pos], 0..K-1
            pred_reg = bbox_pred_i[pos_inds, cls_inds]           # [num_pos, 4]

            reg_loss = F.smooth_l1_loss(pred_reg, targets[pos_inds], reduction="mean")
        else:
            reg_loss = bbox_pred_i.sum() * 0.0

        all_cls_loss.append(cls_loss)
        all_reg_loss.append(reg_loss)

    classification_loss = torch.stack(all_cls_loss).mean()
    bbox_reg_loss = torch.stack(all_reg_loss).mean()

    return classification_loss + lambda_reg * bbox_reg_loss


if __name__ == '__main__':
    class_logits = [torch.randn(2, 7), torch.randn(3, 7)]
    bbox_pred = [torch.randn(2, 24), torch.randn(3, 24)]
    gt_labels = [torch.randint(1, 7, (2, )), torch.randint(1, 7, (3, ))]
    gt_boxes = [torch.randn(2, 4), torch.randn(3, 4)]
    num_classes = 6
    loss = fast_rcnn_loss(class_logits, bbox_pred, gt_labels, gt_boxes, num_classes)
    print(loss)