import torch
import numpy as np

def generate_anchors(feature_map_size, strides, image_size, scales=(64, 128, 256), ratios=(0.5, 1, 2), device='cpu'):
    H, W = feature_map_size
    stride_H, stride_W = strides
    anchors = []
    for i in range(H):
        for j in range(W):
            ctr_x = j * stride_W + stride_W / 2
            ctr_y = i * stride_H + stride_H / 2
            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    x1 = ctr_x - w / 2
                    y1 = ctr_y - h / 2
                    x2 = ctr_x + w / 2
                    y2 = ctr_y + h / 2
                    anchors.append([x1, y1, x2, y2])
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = clip_anchors_to_image(anchors, image_size)
    return anchors



