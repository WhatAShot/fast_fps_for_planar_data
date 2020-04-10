import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import math
import warnings
from matplotlib import pyplot as plt
from functools import wraps


def runtime(func):
    """runtime decorators"""
    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time()
        f = func(*args, **kwargs)     # 原函数
        end = time()
        print("{}: {}s".format(func.__name__, end-start))
        return f
    return wrapper


def density_check(w, h, n_points):
    if w * h > n_points * 5:
        return 'good'
    else:
        return 'too dense'


@runtime
def direct_fps(tensor, n_points):
    device = tensor.device
    (B, C, W, H) = tensor.shape
    centroids = fps(B, W, H, n_points, device)
    coord = torch.stack([centroids % W, centroids // W], 1).transpose_(1, 2)
    return coord


@runtime
def fast_fps(tensor, n_points):
    """
    :param tensor:  [B, C, W, H]
    :param n_points:  the number of points
    :return: sampled point indices
    Assume: 2^2 * 2^2 tensor for one point sampling is considerable
    """

    (B, C, W, H) = tensor.shape
    device = tensor.device
    if W == H:
        if density_check(W, H, n_points) == 'too dense':
            warnings.warn('the tensor might be too small for such number of points', DeprecationWarning)

        rate = W * H // n_points  # grid -> point rate
        pool_rate = int(math.log(math.sqrt(rate))) - 1 if int(math.log(math.sqrt(rate))) > 2 else 0

        w, h = W // (2 ** pool_rate), H // (2 ** pool_rate)
        app_centroids = fps(B, w, h, n_points, device)
        app_coord = torch.stack([app_centroids % w, app_centroids // w], 1).transpose_(1, 2)
        shift_row = torch.randint(0, (2 ** pool_rate), (app_centroids.shape[1], 1), dtype=torch.long).repeat(B, 1, 1)
        shift_col = torch.randint(0, (2 ** pool_rate), (app_centroids.shape[1], 1), dtype=torch.long).repeat(B, 1, 1)
        shift = torch.cat([shift_row, shift_col], 2)
        coord = app_coord * (2 ** pool_rate) + shift
        return coord
    else:
        w = int(W)
        h = int(H)
        centroids = fps(B, w, h, n_points, device)
        coord = torch.stack([centroids % w, centroids // w], 1).transpose_(1, 2)
        return coord


def fps(b, w, h, n_point, device):
    """
    :param b: batch size
    :param w: the weight of tensor
    :param h:  the height of tensor
    :param n_point: the number of points
    :param device: device
    :return: sampled point indices
    """
    centroids = torch.zeros((b, n_point), dtype=torch.long).to(device)
    distance = torch.ones(b, w*h, dtype=torch.long).to(device) * 1e10
    farthest = torch.randint(0, w * h, (b,), dtype=torch.long).to(device)

    for i in range(n_point):
        centroids[:, i] = farthest
        tensor_grid = torch.stack(torch.meshgrid([torch.arange(w, dtype=torch.long),
                                                  torch.arange(h, dtype=torch.long)])).transpose_(0, 2).contiguous().view(w * h, 2).repeat(b, 1, 1)

        centroid = torch.stack([farthest % w, farthest // w]).transpose_(0, 1).view(b, 1, 2)
        dist = torch.sum((tensor_grid - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def fps_3d(xyz, npoint):
    """
    From : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def vis(tensor, coord, tag):
    im = torch.zeros_like(tensor.squeeze()) + 1.
    for j in range(coord.shape[1]):
        row_index = coord[0, j, 0].item()
        col_index = coord[0, j, 1].item()
        im[row_index, col_index] = 0
    plt.imsave(tag + '.jpg', im)
    return None


if __name__ == '__main__':

    image = torch.randint(0, 1, (1, 1, 1024, 1024))

    coords_fast_fps = fast_fps(image, 1024)
    coords_direct_fps = direct_fps(image, 1024)

    vis(image, coords_fast_fps, 'fast_fps')
    vis(image, coords_direct_fps, 'direct_fps')
