import torch

from utils.compute_iou import compute_iou

def assign_anchors(anchors, gt_boxes, positive_iou=0.7, negative_iou=0.3):
    """
    anchors: tensor, shape [num_anchors, 4],  (x1, y1, x2, y2)
    gt_boxes: tensor, shape [num_gt, 4],  (x1, y1, x2, y2)
    positive_iou: float, Anchors with an IoU ≥ positive_iou are labeled as positive samples.
    negative_iou: float, Anchors with an IoU < negative_iou are labeled as negative samples.
    return:
        labels: tensor, shape (num_anchors, )
        bbox_targets: tensor, shape (num_anchors, 4)
    """
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    labels = torch.full((num_anchors, ), -1, dtype=torch.long, device=anchors.device)
    bbox_targets = torch.zeros((num_anchors, 4), dtype=torch.float, device=anchors.device)
    if num_gt == 0:
        labels[:] = 0
        return labels, bbox_targets
    iou = compute_iou(anchors, gt_boxes)    # iou shape: [num_anchors, num_gt]
    max_iou, max_idx = iou.max(dim=1)
    labels[max_iou >= positive_iou] = 1
    labels[max_iou < negative_iou] = 0

    best_anchor_for_gt = iou.argmax(dim=0)
    labels[best_anchor_for_gt] = 1
    max_idx[best_anchor_for_gt] = torch.arange(num_gt, device=anchors.device)

    fg_idx = torch.where(labels == 1)[0]
    if fg_idx.numel() > 0:
        a = anchors[fg_idx]
        g = gt_boxes[max_idx[fg_idx]]

        wa = (a[:, 2] - a[:, 0]).clamp(min=0) + 1e-12
        ha = (a[:, 3] - a[:, 1]).clamp(min=0) + 1e-12
        xa = a[:, 0] + 0.5 * wa
        ya = a[:, 1] + 0.5 * ha

        wg = (g[:, 2] - g[:, 0]).clamp(min=0) + 1e-12
        hg = (g[:, 3] - g[:, 1]).clamp(min=0) + 1e-12
        xg = g[:, 0] + 0.5 * wg
        yg = g[:, 1] + 0.5 * hg
        dx = (xg - xa) / wa
        dy = (yg - ya) / ha
        dw = torch.log(wg / wa)
        dh = torch.log(hg / ha)

        bbox_targets[fg_idx] = torch.stack([dx, dy, dw, dh], dim=1)

    return labels, bbox_targets

if __name__ == '__main__':
    anchors = torch.tensor([
        [0., 0., 10., 10.],    # anchor 0
        [0., 0., 20., 20.],    # anchor 1
        [15., 15., 25., 25.],  # anchor 2
        [50., 50., 60., 60.]   # anchor 3  (complete background)
    ])

    gt_boxes = torch.tensor([
        [0., 0., 10., 10.],   # gt 0       (Completely overlaps with anchor 0)
        [14., 14., 26., 26.]  # gt 1       (Has a high IoU with anchor 2)
    ])

    labels, bbox_targets = assign_anchors(anchors, gt_boxes)
    print(labels)
    print(bbox_targets)