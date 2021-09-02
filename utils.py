import torch
import numpy as np


def position_encoding(x):
    assert position_encoding.L_embed is not None, \
        "Please set this parameter before using this function"

    rets = [x]
    for i in range(position_encoding.L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, dim=-1)


position_encoding.L_embed = None


def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(torch.arange(0, W, device=device),
                          torch.arange(0, H, device=device))
    i, j = i.T, j.T

    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal,
                        -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


def trans_t(t, device):
    return torch.FloatTensor([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, t],
                              [0, 0, 0, 1]]).to(device)


def rot_phi(phi, device):
    return torch.FloatTensor([[1, 0, 0, 0],
                              [0, np.cos(phi), -np.sin(phi), 0],
                              [0, np.sin(phi), np.cos(phi), 0],
                              [0, 0, 0, 1]]).to(device)


def rot_theta(th, device):
    return torch.FloatTensor([[np.cos(th), 0, -np.sin(th), 0],
                              [0, 1, 0, 0],
                              [np.sin(th), 0, np.cos(th), 0],
                              [0, 0, 0, 1]]).to(device)


def pose_spherical(theta, phi, radius, device):
    c2w = trans_t(radius, device)
    c2w = rot_phi(phi/180.*np.pi, device) @ c2w
    c2w = rot_theta(theta/180.*np.pi, device) @ c2w
    c2w = torch.FloatTensor([[-1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]]).to(device) @ c2w
    return c2w
