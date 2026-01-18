import torch

from compute_iou import compute_iou
from bbox2delta import bbox2delta


def assign_proposals_to_gt(proposals, gt_boxes, gt_labels, iou_threshold=0.1):
    """
    proposals shape: [num_proposals, 4], (x1, y1, x2, y2)
    gt_boxes shape: [num_gt_boxes, 4], (x1, y1, x2, y2)
    gt_labels shape: [num_gt_boxes], The category of each ground truth bounding box
    return:
        labels shape: [num_proposals], bbox_targets shape: [num_proposals, 4]
    """
    num_proposals = proposals.shape[0]
    iou = compute_iou(proposals, gt_boxes)    # [num_proposals, num_gt_boxes]
    max_iou, max_idx = torch.max(iou, dim=1)  # [num_proposals]

    labels = torch.zeros(num_proposals, dtype=torch.long, device=proposals.device)
    bbox_targets = torch.zeros((num_proposals, 4), dtype=torch.float32, device=proposals.device)

    pos_inds = max_iou >= iou_threshold       # [num_proposals]
    labels[pos_inds] = gt_labels[max_idx[pos_inds]]
    bbox_targets[pos_inds] = bbox2delta(proposals[pos_inds], gt_boxes[max_idx[pos_inds]])
    return labels, bbox_targets

if __name__ == "__main__":
    proposals = torch.randn(20, 4)
    gt_boxes = torch.randn(10, 4)
    gt_labels = torch.randint(1, 7, (10,))
    labels, bbox_targets = assign_proposals_to_gt(proposals, gt_boxes, gt_labels)
    print(f"labels shape: {labels.shape}, bbox_targets shape: {bbox_targets.shape}")
