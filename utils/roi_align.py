import torch
import torch.nn.functional as F


def roi_align(features, rois, img_size, output_size=(7, 7)):
    """
    features: [B, 2048, H/32, W/32], The output of the backbone
    rois: List[Tensor[N_i, 4]], (x1, y1, x2, y2) in image coords
    img_size: (H, W), The original image size
    return:
        roi_features: List[Tensor[N_i, 2048, H_out, W_out]]
    """
    B, C, H_f, W_f = features.shape
    H_out, W_out = output_size
    H_img, W_img = img_size

    stride_h = H_img / H_f
    stride_w = W_img / W_f

    roi_features = []

    for j in range(B):
        feats = features[j:j+1]        # [1, C, H_f, W_f]
        boxes = rois[j]                # [N_i, 4]
        sample_roi_features = []

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]

            # Mapped to feature map coordinates
            x1_f = x1 / stride_w
            x2_f = x2 / stride_w
            y1_f = y1 / stride_h
            y2_f = y2 / stride_h

            grid_y = torch.linspace(y1_f, y2_f, H_out, device=features.device)
            grid_x = torch.linspace(x1_f, x2_f, W_out, device=features.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

            # Normalize to [-1, 1]
            grid_x = (grid_x / (W_f - 1)) * 2 - 1
            grid_y = (grid_y / (H_f - 1)) * 2 - 1

            grid = torch.stack((grid_x, grid_y), dim=-1)   # [H_out, W_out, 2]
            grid = grid.unsqueeze(0)                       # [1, H_out, W_out, 2]

            roi_feat = F.grid_sample(feats, grid, mode='bilinear', align_corners=True)  # [1, C, H_out, W_out]

            sample_roi_features.append(roi_feat)

        if len(sample_roi_features) > 0:
            sample_roi_features = torch.cat(sample_roi_features, dim=0)
        else:
            sample_roi_features = torch.zeros((0, C, H_out, W_out), device=features.device)

        roi_features.append(sample_roi_features)

    return roi_features


if __name__ == '__main__':
    features = torch.randn(4, 2048, 2, 2)
    rois = [torch.randn(7, 4), torch.randn(4, 4), torch.randn(5, 4), torch.randn(2, 4)]
    img_size = (64, 64)
    roi_features = roi_align(features, rois, img_size)
    for roi in roi_features:
        print(roi.shape)