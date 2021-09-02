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

view_param = {
    "N_samples": 32,
    "near": 2,
    "far": 6,
}


def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples, device=device).view(1, 1, N_samples)

    if rand:
        z_vals = z_vals + torch.rand(list(rays_o.shape[:-1]) + [N_samples], device=device) * (far-near) / N_samples

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    pts = position_encoding(pts)
    raw = network_fn(pts)

    # Compute opacities and colors)
    sigma_a = F.softplus(raw[..., 3])
    warnings.warn('relu might cause zero gradient for every parameter, so here I change it to softplus\n'
                  'use `sigma_a = F.relu(raw[..., 3])` if it can lead to better performance (but it can\'t)')
    rgb = torch.sigmoid(raw[..., :3])

    # Do volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                      torch.full(z_vals[..., :1].shape, 1e10, device=device)], dim=-1)
    alpha = 1. - torch.exp(-sigma_a * dists)

    weights = alpha * (torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1])

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map


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

    for i in range(N_iterations+1):

        img_i = np.random.randint(images.shape[0])
        target = torch.FloatTensor(images[img_i]).to(device)
        pose = torch.FloatTensor(poses[img_i]).to(device)
        rays_o, rays_d = get_rays(H, W, focal, pose, device)

        rgb, depth, acc = render_rays(model, rays_o, rays_d, rand=True, **view_param)
        loss = torch.mean((rgb - target)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % eval_freq == 0:
            print(i, (time.time() - t) / eval_freq, 'secs per iter')
            t = time.time()

            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, testpose, device)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, **view_param)
            loss = torch.mean((rgb - testimg)**2)
            psnr = -10. * torch.log(loss) / np.log(10.)

            psnrs.append(psnr.cpu().detach().numpy())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb.cpu().detach().numpy())
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.savefig(f'checkpoint/iteration_{i}.png')
            plt.close()

    print('Done')

    frames = []
    for th in tqdm.tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4., device)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4], device)
        rgb, _, _ = render_rays(model, rays_o, rays_d, **view_param)
        rgb = torch.clip(255*rgb, 0, 255).cpu().detach().numpy().astype(np.uint8)
        frames.append(rgb)

    import imageio
    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)

    from IPython.display import HTML
    from base64 import b64encode
    mp4 = open('video.mp4', 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls autoplay loop>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
