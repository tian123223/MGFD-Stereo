import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

def normalize_coords(grid):
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1
    grid = grid.permute((0, 2, 3, 1))
    return grid


def meshgrid(img, homogeneous=False):

    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)
    grid = grid.unsqueeze(0).expand(b, 2, h, w)

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)
        grid = torch.cat((grid, ones), dim=1)
        assert grid.size(1) == 3
    return grid


def disp_warp(img, disp, padding_mode='border'):

    grid = meshgrid(img)
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode,align_corners=False)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros',align_corners=False)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask


