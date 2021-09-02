import os
import time
import imageio
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from IPython.display import HTML
from base64 import b64encode

import model
from utils import position_encoding, get_rays, pose_spherical

os.environ['CUDA_VISIBLE_DEVICES'] = "9"
os.makedirs('checkpoint', exist_ok=True)

device = torch.device('cuda')
position_encoding.L_embed = 6
eval_freq = 25
N_iterations = 1000
accumulate_iter = 4

view_param = {
    "N_samples": 64,
    "near": 2,
    "far": 6,
}



@torch.no_grad()
def eval_one_view(network_fn, rays_o, rays_d, near, far, N_samples, chunk=8192):
    H, W, C = rays_o.shape

    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples, device=device).view(1, N_samples)
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                      torch.full(z_vals[..., :1].shape, 1e10, device=device)], dim=-1)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts = pts.view(H*W, N_samples, C)

    rgbs = []
    for i in range(0, pts.shape[0], chunk):
        points = position_encoding(pts[i: i+chunk])

        raw = network_fn(points)

        # Compute opacities and colors)
        sigma_a = F.softplus(raw[..., 3])
        rgb = torch.sigmoid(raw[..., :3])

        # Do volume rendering
        alpha = 1. - torch.exp(-sigma_a * dists)

        # alpha = [a, b, c, d]  >>>  output = [a, b(1-a), c(1-a)(1-b), d(1-a)(1-b)(1-c)]
        weights = alpha * (torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1])
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

        rgbs.append(rgb_map)

    image = torch.cat(rgbs, dim=0).view(H, W, 3)
    return image


def train_one_view(network_fn, rays_o, rays_d, target, near, far, N_samples, chunk=8192):
    H, W, C = rays_o.shape

    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples, device=device).view(1, 1, N_samples)
    # add noise
    z_vals = z_vals + torch.rand(H, W, N_samples, device=device) * (far-near) / N_samples

    pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_vals[..., :, None]

    pts = pts.view(H*W, N_samples, C)
    target = target.view(H*W, 3)
    z_vals = z_vals.view(H * W, N_samples)

    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                      torch.full(z_vals[..., :1].shape, 1e10, device=device)], dim=-1)

    for i in range(0, pts.shape[0], chunk):
        points = position_encoding(pts[i: i+chunk])

        raw = network_fn(points)

        # Compute opacities and colors)
        sigma_a = F.softplus(raw[..., 3])
        rgb = torch.sigmoid(raw[..., :3])

        # Do volume rendering
        alpha = 1. - torch.exp(-sigma_a * dists[i: i+chunk])

        # alpha = [a, b, c, d]  >>>  output = [a, b(1-a), c(1-a)(1-b), d(1-a)(1-b)(1-c)]
        weights = alpha * (torch.cumprod(torch.cat([torch.ones_like(
            alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1])

        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        loss = torch.sum((rgb_map - target[i:i+chunk]) ** 2)
        loss.backward()

    return


if __name__ == '__main__':

    # Load data
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = float(data['focal'])
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    # Split train / test
    testimg, testpose = images[101], poses[101]
    images = images[:100, :, :, :3]
    poses = poses[:100]

    plt.imshow(testimg)
    plt.savefig('sample.png')

    testimg = torch.FloatTensor(testimg).to(device)
    testpose = torch.FloatTensor(testpose).to(device)

    # Define Model
    model = model.NERF(D=8, W=256, L_embed=position_encoding.L_embed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)

    # initialize meters
    psnrs = []
    iternums = []
    t = time.time()
    tqdm_iter = tqdm.tqdm(range(N_iterations+1))
    for i in tqdm_iter:

        img_i = np.random.randint(images.shape[0])
        target = torch.FloatTensor(images[img_i]).to(device)
        pose = torch.FloatTensor(poses[img_i]).to(device)
        rays_o, rays_d = get_rays(H, W, focal, pose, device)

        if i % accumulate_iter == 0:
            optimizer.zero_grad()

        train_one_view(model, rays_o, rays_d, target, **view_param)

        if (i + 1) % accumulate_iter == 0:
            optimizer.step()

        if i % eval_freq == 0:
            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, testpose, device)
            rgb = eval_one_view(model, rays_o, rays_d, **view_param)
            loss = torch.mean((rgb - testimg)**2)
            psnr = -10. * torch.log(loss) / np.log(10.)

            psnrs.append(psnr.cpu().numpy())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb.cpu().numpy())
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.savefig(f'checkpoint/iteration_{i}.png')
            plt.close()
            tqdm_iter.set_description(f'checkpoint/iteration_{i}.png generated')

    print('Done')

    frames = []
    for th in tqdm.tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4., device)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4], device)
        rgb = eval_one_view(model, rays_o, rays_d, **view_param)
        rgb = torch.clip(255*rgb, 0, 255).cpu().detach().numpy().astype(np.uint8)
        frames.append(rgb)

    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)

    mp4 = open('video.mp4', 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls autoplay loop>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
