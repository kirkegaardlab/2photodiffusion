import pickle
from functools import lru_cache
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn.functional import pad
from tqdm import tqdm


def gen_trajectory_times(X, Y, v, delay):
    T = X / v + Y * (delay + X.max() / v)
    return T


def gen_qt_trajectory(x0, y0, z0, t, d):
    x = x0[:, None] + 0 * t[None, :]
    y = y0[:, None] + 0 * t[None, :]
    z = z0[:, None] + 0 * t[None, :]
    x[:, 1:] += torch.cumsum(torch.sqrt(2 * d * (t[1:] - t[:-1]))[None] *
                             torch.randn((x.shape[0], x.shape[1] - 1), device=x.device), dim=1)
    y[:, 1:] += torch.cumsum(torch.sqrt(2 * d * (t[1:] - t[:-1]))[None] *
                             torch.randn((x.shape[0], x.shape[1] - 1), device=x.device), dim=1)
    z[:, 1:] += torch.cumsum(torch.sqrt(2 * d * (t[1:] - t[:-1]))[None] *
                             torch.randn((x.shape[0], x.shape[1] - 1), device=x.device), dim=1)
    return x, y, z


def image_mean(X, Y, x, y, z, Nbg, Nqd, sxy, sz, M):
    mask = 1 * (torch.arange(x.shape[0], device=x.device) <= M)
    idx = X + X.shape[1] * Y

    xX = x[:, idx]
    yY = y[:, idx]
    zZ = z[:, idx]
    X = X[None]
    Y = Y[None]

    return Nbg + torch.sum(mask[:, None, None] * Nqd / (2 * torch.pi * sxy**2) * torch.exp(-((X - xX)**2 + (Y - yY)**2) / (2 * sxy**2)) \
                           * torch.exp(-zZ**2 / (2 * sz**2)), dim=0)


def image_transform(im, Nbg, fac):
    std = 0.5 + 2 * torch.rand(1, device=im.device)
    sub = Nbg - 3 + 2 * torch.randn(1, device=im.device)
    im = im + std * torch.randn_like(im)
    im = -sub + im
    im = fac * im
    return im


def make_dataset(steps=5, N=150_000, device='cuda',
                 dx=0.15, line_speed=125000, line_delay=0.001348, reset_time=0.033,
                 overwrite_d=None, all_M=False):
    Nbg = torch.abs(100.0 + 20 * torch.randn(N, device=device))
    Nqd = torch.abs(600.0 + 200 * torch.randn(N, device=device))
    sxy = 0.21 + 0.05 * torch.randn(N, device=device)  # um
    sz = 1.36 + 0.2 * torch.randn(N, device=device)  # um

    cam_speed = line_speed * 1e-6  # px / us
    line_delay = line_delay * 1e6  # us
    reset_time = reset_time * 1e6  # us

    res = dx  # um / px
    time_scale = 1e-6  # us

    X, Y, t, xx, yy = calc_txy(cam_speed, device, line_delay)
    frame_dt = (t.max() + reset_time) * time_scale

    d_min = 0.0  # um^2 / s
    d_max = 1.5  # um^2 / s
    d = d_min + (d_max - d_min) * torch.rand(N, device=device)

    if overwrite_d is not None:
        d[:] = overwrite_d

    M_max = 98
    M = torch.tensor(np.random.exponential(scale=0.25 * M_max, size=N), dtype=torch.int64, device=device)
    M[M >= M_max] = M_max - 1
    M2 = torch.tensor(np.random.randint(0, M_max, size=N), dtype=torch.int64, device=device)
    M_mask = torch.rand(N, device=device) > 0.5
    M[M_mask] = M2[M_mask]
    if all_M:
        M[:] = M_max

    spread = 2.0 + 0.5 * torch.randn(N, device=device)
    x0, y0 = spread[None, :, None] * torch.randn((2, N, M_max), device=device)
    x0 *= xx.max()
    y0 *= yy.max()
    x0 += xx.max() / 2
    y0 += yy.max() / 2
    z0 = spread[:, None] * 0.5 / res * torch.randn((N, M_max), device=device)

    t0 = torch.tensor(np.random.randint(0, steps, size=N), device=device)

    data = {
        'im': [],
        'path': [],
        'd': d,
        'M': M,
        't': frame_dt * torch.arange(steps, device=device),
        'params': dict(Nbg=Nbg, Nqd=Nqd, sxy=sxy, sz=sz, reset_time=reset_time,
                       cam_speed=cam_speed, line_delay=line_delay, res=res, time_scale=time_scale,
                       d_min=d_min, d_max=d_max)
    }

    sxy = sxy / res
    sz = sz / res
    d = d / res**2 * time_scale  # px^2 / us

    x0_save = x0.clone()
    y0_save = y0.clone()
    z0_save = z0.clone()

    transform_fac = torch.abs(1 + 0.1 * torch.randn(N, device=device))

    for step in range(steps):
        x, y, z = torch.vmap(gen_qt_trajectory, in_dims=(0, 0, 0, None, 0), randomness='different')(x0, y0, z0, t, d)
        mu = torch.vmap(image_mean, in_dims=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))(X, Y, x, y, z, Nbg, Nqd, sxy, sz, M)

        im = torch.poisson(mu)
        im = torch.vmap(image_transform, in_dims=(0, 0, 0), randomness='different')(im, Nbg, transform_fac)

        # Save
        data['im'].append(im / 100)
        p = torch.cat((x[:, :, None], y[:, :, None], z[:, :, None]), dim=2)

        remove_middle = True
        if remove_middle:
            p = p[..., p.shape[-1] // 2][..., None]

        data['path'].append(p)

        # Prepare for next image:
        x0 = x[:, :, -1]
        y0 = y[:, :, -1]
        z0 = z[:, :, -1]
        mask = (step + 1) == t0
        x0[mask, :] = x0_save[mask, :]
        y0[mask, :] = y0_save[mask, :]
        z0[mask, :] = z0_save[mask, :]

        x0 += torch.randn(x0.shape, device=device) * torch.sqrt(2 * d * reset_time)[:, None]
        y0 += torch.randn(x0.shape, device=device) * torch.sqrt(2 * d * reset_time)[:, None]
        z0 += torch.randn(x0.shape, device=device) * torch.sqrt(2 * d * reset_time)[:, None]


    im = torch.stack(data['im'])
    im = torch.concatenate((im, torch.zeros((im.shape[0], im.shape[1], 20, 4), dtype=im.dtype, device=device)), dim=3)
    im = torch.concatenate((im, torch.zeros((im.shape[0], im.shape[1], 4, 24), dtype=im.dtype, device=device)), dim=2)
    data['im'] = im
    data['path'] = torch.stack(data['path'])


    reorder = torch.repeat_interleave(torch.arange(steps)[:, None], len(t0), dim=1)
    for i in range(len(t0)):
        reorder[:t0[i], i] = torch.flip(reorder[:t0[i], i], dims=(0, ))
        data['im'][:, i] = data['im'][reorder[:, i], i]
        data['path'][:, i] = data['path'][reorder[:, i], i]

    data['im'] = torch.transpose(data['im'], 0, 1)
    data['path'] = torch.transpose(data['path'], 0, 1)


    return data


@lru_cache
def calc_txy(cam_speed, device, line_delay):
    xx = torch.arange(20, device=device)
    yy = torch.arange(20, device=device)
    X, Y = torch.meshgrid(xx, yy, indexing='xy')
    T = gen_trajectory_times(X, Y, v=cam_speed, delay=line_delay)
    t = torch.sort(torch.unique(T))[0]
    return X, Y, t, xx, yy


def save_frames():
    data = make_dataset(N=1, steps=400, device='cpu', overwrite_d=1.5, all_M=True)
    all_ims = data['im']

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from skimage.transform import rescale
    from skimage.io import imsave
    k = 0
    for i in tqdm(range(all_ims.shape[1])):
        for j in range(all_ims.shape[0]):
            im = all_ims[j, i, :20, :20]
            cmap = plt.get_cmap('magma')
            ims = np.clip(0.5 * (im + 0.5), 0, 1)
            ims = cmap(ims)[..., :3]
            ims = (255 * ims)
            im = rescale(ims, (10, 10, 1), order=0).astype(np.uint8)
            imsave(f'frames/{k:05d}.png', im)
            k += 1



if __name__ == '__main__':
    save_frames()
