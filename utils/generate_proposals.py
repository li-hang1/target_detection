import torch.nn.functional as F

from nms import nms
from apply_deltas_to_anchors import apply_deltas_to_anchors
from clip_anchors_to_image import clip_anchors_to_image

def generate_proposals(cls_logits, bbox_pred, anchors, img_size, post_nms_top_n=300, nms_threshold=0.7):
    """
    cls_logits: shape [B, 2 * k, H/stride_H, W/stride_W], The output of the RPN network
    bbox_pred: shape [B, 4 * k, H/stride_H, W/stride_W], The output of the RPN network
    anchors: shape [num_anchors, 4]
    img_size: (H, W), The original image size
    return:
        List[Tensor[N_i, 4]], N_i represents the number of remaining anchors.
    """
    B, _, H, W = cls_logits.shape
    cls_probs = F.softmax(cls_logits.permute(0, 2, 3, 1).reshape(B, -1, 2), dim=-1)[:, :, 1]
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
    all_proposals = []
    for i in range(B):
        scores = cls_probs[i]
        deltas = bbox_pred[i]
        proposals = apply_deltas_to_anchors(anchors, deltas)
        proposals = clip_anchors_to_image(proposals, img_size)
        keep = nms(proposals, scores, iou_threshold=nms_threshold)
        keep = keep[:post_nms_top_n]
        proposals_keep = proposals[keep]
        all_proposals.append(proposals_keep)
    return all_proposals


if __name__ == '__main__':
    import torch
    cls_logits = torch.randn(4, 4, 2, 2)
    bbox_pred = torch.randn(4, 8, 2, 2)
    anchors = torch.randn(8, 4)
    img_size = (64, 64)
    proposals = generate_proposals(cls_logits, bbox_pred, anchors, img_size)
    for proposal in proposals:
        print(proposal.shape)
