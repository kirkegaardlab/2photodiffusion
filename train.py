import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


def transform(x):
    m1 = torch.randint(0, 2, size=(x.shape[0],), device=x.device)[:, None, None, None]
    x = m1 * x + (1 - m1) * torch.flip(x, dims=(-1,))

    m1 = torch.randint(0, 2, size=(x.shape[0],), device=x.device)[:, None, None, None]
    x = m1 * x + (1 - m1) * torch.flip(x, dims=(-2,))

    m1 = torch.randint(0, 2, size=(x.shape[0],), device=x.device)[:, None, None, None]
    x = m1 * x + (1 - m1) * torch.transpose(x, -1, -2)

    return x


def train(model, data_generator, steps=15, device='cpu', what='L2', aux_model=None, save=None,
          lr=5e-4, batch_size=None, max_batch_use=None, warmup_lr=None, fast_steps=0, fast_dataset=None):
    if what == 'sigma':
        assert aux_model is not None


    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    if aux_model is not None:
        aux_model.to(device)

    if save is not None:
        writer = SummaryWriter(f'logs/{save}')
    else:
        writer = None

    acc_tuple = (0, 0, 0)
    pbar = tqdm(range(steps))
    step = 0

    while step < steps:
        if warmup_lr is not None:
            if step == 0:
                for g in opt.param_groups:
                    g['lr'] = warmup_lr
            elif step == 3000:
                for g in opt.param_groups:
                    g['lr'] = lr

        x, y = data_generator()

        if batch_size is None:
            dataloader = [(x, y)]
        else:
            dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, drop_last=False, shuffle=True)

        for count, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)

            step += 1
            pbar.update(1)

            if max_batch_use is not None and count == max_batch_use:
                break

            rr = model(xx)
            if what == 'L2':
                loss = torch.mean((rr - yy) ** 2)
                with torch.no_grad():
                    L1_loss = torch.mean(torch.abs(rr - yy))
            elif what == 'sigma':
                with torch.no_grad():
                    mu = aux_model(xx)
                sigma = 1e-7 + rr
                loss = torch.mean((mu - yy) ** 2 / (2 * sigma**2) + torch.log(sigma))

                with torch.no_grad():
                    L1_loss = torch.mean(torch.abs(torch.abs(mu - yy) - sigma))
            elif what == 'L2_sigma':
                mu = rr[..., 0][..., None]
                sigma = 0.02 + rr[..., 1][..., None]
                if False and step < steps / 2:
                    loss = torch.mean((mu - yy) ** 2)
                else:
                    loss = torch.mean((mu - yy) ** 2 / (2 * sigma ** 2) + torch.log(sigma))

                with torch.no_grad():
                    L1_loss = torch.mean(torch.abs(mu - yy))
            else:
                raise AssertionError(f"No such train method `{what}`")

            loss.backward()
            opt.step()
            opt.zero_grad()
            acc_tuple = (acc_tuple[0] + 1, acc_tuple[1] + loss.detach(), acc_tuple[2] + L1_loss)

            if step % (1 + (steps // 1000)) == 0:
                print(flush=True)
                print(f'      train loss = {(acc_tuple[1] / acc_tuple[0]):0.7f}'
                      f'      L1 loss = {(acc_tuple[2] / acc_tuple[0]):0.3f}.')
                if save:
                    writer.add_scalar('loss', acc_tuple[1] / acc_tuple[0], step)
                    writer.add_scalar('L1_loss', acc_tuple[2] / acc_tuple[0], step)
                acc_tuple = (0, 0, 0)

            if step % (1 + (steps // 20)) == 0:
                if save is not None:
                    model.eval()
                    model.to('cpu')
                    torch.save(model.state_dict(), f'store/{save}.p')
                    model.to(device)
                    model.train()
                    print('saved.')

    model.eval()
    if save is not None:
        model.to('cpu')
        torch.save(model.state_dict(), f'store/{save}.p')
        model.to(device)
        print('saved.')


def train_aa():
    print('=== TRAINING AA NETWORK ===')
    from models import Autoencoder
    from sim import make_dataset

    aa = Autoencoder(base_channel_size=32, latent_dim=128)
    try:
        aa.load_state_dict(torch.load('store/aa.p'))
    except FileNotFoundError:
        print('Warning: Training AA from scratch')
    aa.to('cuda')

    def data_generator():
        T = int(np.random.randint(5, 400))
        N = int(1e4 / T)
        x = make_dataset(steps=T, N=N)['im'].reshape(-1, 1, 24, 24)
        return x, x

    train(aa, data_generator, steps=5_000, device='cuda', what='L2', save='aa', lr=5e-4,
          batch_size=512, max_batch_use=3)


def train_d_predictor():
    print('=== TRAINING D NETWORK ===')
    from models import Autoencoder, AATCNN
    from sim import make_dataset

    aa = Autoencoder(base_channel_size=32, latent_dim=128)
    aa.load_state_dict(torch.load('store/aa.p'))
    aa.to('cuda')
    model = AATCNN(aa.encoder, factor=1.0, n_classes=2)
    try:
        model.load_state_dict(torch.load('store/aa_encoder_and_sigma.p'))
    except (FileNotFoundError, RuntimeError):
        print('Warning: Training D estimator from scratch')
    model.to('cuda')

    def data_generator():
        data = make_dataset(steps=400, N=50)
        return data['im'], data['d'][:, None]


    train(model, data_generator, steps=1_000, device='cuda', what='L2_sigma', save='aa_encoder_and_sigma', lr=0.01,
          warmup_lr=None, fast_steps=0)



if __name__ == '__main__':
    import os
    if not os.path.exists('store'):
        os.makedirs('store')

    train_aa()
    train_d_predictor()

