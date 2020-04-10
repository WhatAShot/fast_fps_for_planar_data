import torch
from time import time
import math
import warnings
from matplotlib import pyplot as plt
from functools import wraps


def runtime(func):
    """runtime decorators"""
    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time()
        f = func(*args, **kwargs)
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
    if density_check(W, H, n_points) == 'too dense':
        warnings.warn('the tensor might be too small for such number of points', DeprecationWarning)
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


@runtime
def direct_online_fps(tensor, pre_sampled_index, n_points, output_type='all'):
    """
    :param tensor: [B, W * H, 2] or [B, C, W, H]
    :param pre_sampled_index: [B, S] or [B, S, 2]
    :param n_points: int
    :param output_type: 'all' or 'new'
    :return:
    """

    device = tensor.device
    if len(tensor.shape) == 4 and len(pre_sampled_index.shape) == 3:
        (B, C, W, H) = tensor.shape
        if density_check(W, H, n_points + pre_sampled_index.shape[1]) == 'too dense':
            warnings.warn('the tensor might be too small for such number of points', DeprecationWarning)
        flatten_tensor = torch.stack(torch.meshgrid([torch.arange(W, dtype=torch.long),
                                                     torch.arange(H, dtype=torch.long)])).transpose_(0, 2).contiguous().view(W * H, 2).repeat(B, 1, 1).to(device)
        S = pre_sampled_index.shape[1]
        index_bs = torch.zeros((B, S), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        point_indices = torch.arange(S, dtype=torch.long).to(device)
        index_bs[batch_indices, point_indices] = pre_sampled_index[batch_indices, point_indices, 1] * W \
                                                 + pre_sampled_index[batch_indices, point_indices, 0]

    elif len(tensor.shape) == 3 and len(pre_sampled_index.shape) == 2:
        W = int(math.sqrt(tensor.shape[1]))
        flatten_tensor = tensor
        index_bs = pre_sampled_index
    else:
        print('tensor shape:', tensor.shape)
        print('pre_sampled_index.shape', pre_sampled_index.shape)
        raise ValueError('tensor.shape == [B, C, W, H] or [B, W * H, 2]; pre_sampled_index.shape == [B, S, 2] or [B, S]')

    centroids = online_fps(index_bs, flatten_tensor, n_points, output_type)
    coord = torch.stack([centroids % W, centroids // W], 1).transpose_(1, 2)
    return coord


@runtime
def fast_online_fps(tensor, pre_sampled_index, n_points, output_type='all'):
    """
        :param tensor: [B, C, W, H]
        :param pre_sampled_index: [B, S, 2]
        :param n_points: int
        :param output_type: 'all' or 'new'
        :return:
        """

    device = tensor.device
    (B, C, W, H) = tensor.shape
    S = pre_sampled_index.shape[1]
    if density_check(W, H, n_points + S) == 'too dense':
        warnings.warn('the tensor might be too small for such number of points', DeprecationWarning)

    rate = W * H // (n_points + S)  # grid -> point rate
    pool_rate = int(math.log(math.sqrt(rate))) - 1 if int(math.log(math.sqrt(rate))) > 2 else 0

    w, h = W // (2 ** pool_rate), H // (2 ** pool_rate)

    flatten_tensor = torch.stack(torch.meshgrid([torch.arange(w, dtype=torch.long),
        torch.arange(h, dtype=torch.long)])).transpose_(0, 2).contiguous().view(w * h, 2).repeat(B, 1, 1).to(device)

    index_bs = torch.zeros((B, S), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    point_indices = torch.arange(S, dtype=torch.long).to(device)
    index_bs[batch_indices, point_indices] = pre_sampled_index[batch_indices, point_indices, 1] \
                                             // (2 ** pool_rate) * w + pre_sampled_index[
                                                 batch_indices, point_indices, 0] // (2 ** pool_rate)

    centroids_new = online_fps(index_bs, flatten_tensor, n_points, 'new')
    coord_new = torch.stack([centroids_new % w, centroids_new // w], 1).transpose_(1, 2)
    shift_row = torch.randint(0, (2 ** pool_rate), (centroids_new.shape[1], 1), dtype=torch.long).repeat(B, 1, 1)
    shift_col = torch.randint(0, (2 ** pool_rate), (centroids_new.shape[1], 1), dtype=torch.long).repeat(B, 1, 1)
    shift = torch.cat([shift_row, shift_col], 2)
    coord_new = coord_new * (2 ** pool_rate) + shift
    coord_all = torch.cat([pre_sampled_index, coord_new], 1)
    return coord_all


def index_points(points, idx):
    """
    From : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def online_fps(sampled_index, flatten_tensor, n_points, output='all'):
    """
    :param sampled_index: [B, S]
    :param flatten_tensor: [B, W * H, C]
    :return: centroids
    """
    assert output in ['all', 'new']
    B, N, C = flatten_tensor.shape
    S = sampled_index.shape[1]
    device = flatten_tensor.device
    sampled_tensor = index_points(flatten_tensor, sampled_index)
    sampled_tensor_exp = sampled_tensor.unsqueeze(1).expand(B, N, S, C)   # [B, S] -> [B, S, C] -> [B, N, S, C]
    flatten_tensor_exp = flatten_tensor.unsqueeze(2).expand(B, N, S, C)   # [B, N, C] -> [B, N, S, C]
    dist_NS = torch.sum((flatten_tensor_exp - sampled_tensor_exp) ** 2, -1)
    distance = torch.min(dist_NS, -1)[0]
    farthest = torch.max(distance, -1)[1]
    centroids = torch.zeros(B, n_points, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_points):
        centroid = flatten_tensor[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((flatten_tensor - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        centroids[:, i] = farthest
    if output == 'all':
        return torch.cat([sampled_index, centroids], 1)
    else:
        return centroids


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
                                                  torch.arange(h, dtype=torch.long)])).transpose_(0, 2).contiguous().view(w * h, 2).repeat(b, 1, 1).to(device)

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
    # distance = torch.ones(B, N).to(device) * 1e10
    distance = torch.ones(B, N, dtype=torch.long).to(device) * 1e10
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
    """
    :param tensor: image
    :param coord: [B, S, 2]
    :param tag: name for save
    :return: None
    """
    im = torch.zeros_like(tensor.squeeze()) + 1.
    for j in range(coord.shape[1]):
        row_index = coord[0, j, 1].item()
        col_index = coord[0, j, 0].item()
        im[row_index, col_index] = 0
    plt.imsave(tag + '.jpg', im)
    return None


if __name__ == '__main__':

    def test_fps():
        image = torch.randint(0, 1, (1, 1, 1024, 1024))

        coords_fast_fps = fast_fps(image, 1024)
        coords_direct_fps = direct_fps(image, 1024)

        vis(image, coords_fast_fps, 'fast_fps')
        vis(image, coords_direct_fps, 'direct_fps')

    def test_online_fps1():

        b, w, h = 1, 64, 64
        flatten_image = torch.stack(torch.meshgrid([torch.arange(w, dtype=torch.long),
                 torch.arange(h, dtype=torch.long)])).transpose_(0, 2).contiguous().view(w * h, 2).repeat(b, 1, 1)
        pre_sampled_index = fps_3d(flatten_image, 100)
        coord = direct_online_fps(flatten_image, pre_sampled_index, 100, 'all')
        vis(torch.randint(0, 1, (b, 1, w, h)), coord, 'online_fps')

    def test_online_fps2():
        image = torch.randint(0, 1, (1, 1, 1024, 1024))
        coords_fast_fps = fast_fps(image, 1024)
        coord = direct_online_fps(image, coords_fast_fps, 100, 'all')
        vis(image, coord, 'online_fps_2')


    def test_fast_online_fps():
        image = torch.randint(0, 1, (1, 1, 1024, 1024))
        coords_fast_fps = fast_fps(image, 1024)
        vis(image, coords_fast_fps, 'pre_online_sampled')
        coord = fast_online_fps(image, coords_fast_fps, 1024, 'all')
        vis(image, coord, 'fast_online_fps')

    # test_fps()
    # test_online_fps1()
    # test_online_fps2()
    test_fast_online_fps()


