import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import model

os.environ['CUDA_VISIBLE_DEVICES'] = "9"
os.makedirs('iteration', exist_ok=True)

device = torch.device('cuda')
L_embed = 6
N_samples = 30
N_iters = 1000
psnrs = []
iternums = []
i_plot = 25

def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, dim=-1)


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(0, W, device=device), torch.arange(0, H, device=device))
    i, j = i.T, j.T

    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], dim=-1)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return rays_o, rays_d


def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples, device=device).view(1, 1, N_samples)
    if rand:
        z_vals = z_vals + torch.rand(list(rays_o.shape[:-1]) + [N_samples], device=device) * (far-near) / N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    pts = posenc(pts)
    raw = network_fn(pts)

    # Compute opacities and colors
    # sigma_a = F.relu(raw[..., 3])
    sigma_a = F.softplus(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3]) 

    # Do volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.full(z_vals[..., :1].shape, 1e10, device=device)], dim=-1)
    alpha = 1. - torch.exp(-sigma_a * dists)  

    weights = alpha * (torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1])
    # front_alpha = torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha[..., :-1] + 1e-10], dim=-1)
    # weights = alpha * torch.cumprod(front_alpha, dim=-1)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1) 
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map


trans_t = lambda t : torch.FloatTensor([[1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,t],
                                        [0,0,0,1]]).to(device)

rot_phi = lambda phi : torch.FloatTensor([[1,0,0,0],
                                          [0,np.cos(phi),-np.sin(phi),0],
                                          [0,np.sin(phi), np.cos(phi),0],
                                          [0,0,0,1]]).to(device)

rot_theta = lambda th : torch.FloatTensor([[np.cos(th),0,-np.sin(th),0],
                                           [0,1,0,0],
                                           [np.sin(th),0, np.cos(th),0],
                                           [0,0,0,1]]).to(device)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.FloatTensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).to(device) @ c2w
    return c2w



if __name__ == '__main__':

    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = float(data['focal'])
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    testimg, testpose = images[101], poses[101]
    images = images[:100,...,:3]
    poses = poses[:100]

    plt.imshow(testimg)
    plt.savefig('sample.png')

    testimg = torch.FloatTensor(testimg).to(device)
    testpose = torch.FloatTensor(testpose).to(device)

    model = model.NERF(D=8, W=256, L_embed=L_embed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)

    t = time.time()
    for i in range(N_iters+1):
        
        img_i = np.random.randint(images.shape[0])
        target = torch.FloatTensor(images[img_i]).to(device)
        pose = torch.FloatTensor(poses[img_i]).to(device)
        rays_o, rays_d = get_rays(H, W, focal, pose)

        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
        loss = torch.mean((rgb - target)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % i_plot==0:
            print(i, (time.time() - t) / i_plot, 'secs per iter')
            t = time.time()
            
            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, testpose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = torch.mean((rgb - testimg)**2)
            psnr = -10. * torch.log(loss) / np.log(10.)

            psnrs.append(psnr.cpu().detach().numpy())
            iternums.append(i)
            
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            plt.imshow(rgb.cpu().detach().numpy())
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.savefig(f'iteration/{i}.png')
            plt.close()

    print('Done')


    frames = []
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
        rgb, _, _ = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        rgb = torch.clip(255*rgb,0,255).cpu().detach().numpy().astype(np.uint8)
        frames.append(rgb)

    import imageio
    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)


    from IPython.display import HTML
    from base64 import b64encode
    mp4 = open('video.mp4','rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls autoplay loop>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
