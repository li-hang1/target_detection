import torch

from .compute_iou import compute_iou
from .bbox2delta import bbox2delta


def assign_proposals_to_gt(proposals, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    proposals shape: [num_proposals, 4], (x1, y1, x2, y2)
    gt_boxes shape: [num_gt_boxes, 4], (x1, y1, x2, y2)
    gt_labels shape: [num_gt_boxes], The category of each ground truth bounding box, (1...K)
    iou_threshold: float(0~1), Only proposals with iou greater than iou_threshold will be classified as positive samples.
    return:
        labels shape: [num_proposals], bbox_targets shape: [num_proposals, 4]
    """
    num_proposals, num_gt_boxes = proposals.shape[0], gt_boxes.shape[0]
    iou = compute_iou(proposals, gt_boxes)  # [num_proposals, num_gt_boxes]
    max_iou, max_idx = iou.max(dim=1)       # [num_proposals]

    labels = torch.full((num_proposals,), -1, dtype=torch.long, device=proposals.device)
    bbox_targets = torch.zeros((num_proposals, 4), device=proposals.device)

    pos_inds = max_iou >= iou_threshold     # [num_proposals]
    labels[pos_inds] = gt_labels[max_idx[pos_inds]]
    bbox_targets[pos_inds] = bbox2delta(proposals[pos_inds], gt_boxes[max_idx[pos_inds]])

    gt_max_iou, gt_argmax = iou.max(dim=0)  # [num_gt_boxes]

    for gt_id in range(num_gt_boxes):
        prop_id = gt_argmax[gt_id]
        labels[prop_id] = gt_labels[gt_id]
        bbox_targets[prop_id] = bbox2delta(proposals[prop_id:prop_id+1], gt_boxes[gt_id:gt_id+1])

    return labels, bbox_targets


if __name__ == "__main__":
    proposals = torch.randn(20, 4)
    gt_boxes = torch.randn(10, 4)
    gt_labels = torch.randint(1, 7, (10,))
    labels, bbox_targets = assign_proposals_to_gt(proposals, gt_boxes, gt_labels)
    print(f"labels shape: {labels.shape}, bbox_targets shape: {bbox_targets.shape}")

