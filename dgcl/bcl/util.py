import os

import numpy as np
import torch
import cv2


def mask2edge(mask, num_class):
    # 创建输出目录（如果不存在）
    # os.makedirs('./mask2edge/', exist_ok=True)
    mask = mask.cpu()  # b,h,w
    target_boundary = np.zeros_like(mask, dtype=np.uint8)
    for i in range(mask.shape[0]):
        label_image = (mask[i]).to(torch.uint8).numpy()
        edges = np.zeros_like(label_image, dtype=np.uint8)
        for class_id in range(0, num_class):
            class_mask = np.where(label_image == class_id, 255, 0).astype(np.uint8)
            class_edges = cv2.Canny(class_mask, 0, 255)
            edges = np.maximum(edges, class_edges)
        target_boundary[i] = edges
        # # save mask and bound
        # cv2.imwrite(f"./mask2edge/edge_{i}.png", edges)
        # cv2.imwrite(f"./mask2edge/mask_{i}.png", label_image * 50)

    target_boundary[target_boundary == 255] = 1
    target_boundary[target_boundary == 0] = 0
    target_boundary = torch.tensor(target_boundary)

    return target_boundary


def one_hot(input_tensor, n_classes):
    """
    :param input_tensor: shape (b,1,h,w)
    """
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()