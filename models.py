import torch
from torch import nn
from resnet1d import ResNet1D


class Encoder(nn.Module):
    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
        )
        self.last = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 3 * 3 * c_hid, latent_dim)
        )

    def forward(self, x):
        x = self.net(x)  # (1, 24 x 24) -> (C, 3, 3)
        x = self.last(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 3 * 3 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 3, 3)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class = Encoder,
                 decoder_class = Decoder,
                 num_input_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class AATCNN(nn.Module):
    def __init__(self, encoder, h=512, n_classes=1, factor=0.1):
        super().__init__()
        encoder.eval()
        self.encoder = [encoder]  # to avoid adding it to the module list
        self.relu = nn.ReLU(inplace=True)
        self.h = h
        self.factor = factor
        self.res = ResNet1D(encoder.latent_dim, h, kernel_size=7, stride=2, n_block=7, groups=1, n_classes=n_classes)

    def forward(self, x):
        with torch.no_grad():
            x = x[:, :, None]
            x = torch.vmap(self.encoder[0], in_dims=1, out_dims=2)(x)

        x = self.res(x)
        x = nn.functional.elu(x) + 1  # enforce positive, softly
        return self.factor * x


if __name__ == '__main__':
    batch_size = 7
    latent = 64
    x_ = torch.randn((400, 1, 24, 24))
    aa = Autoencoder(base_channel_size=32, latent_dim=latent)
    r = aa(x_)
    assert x_.shape == r.shape
    print('aa =', r.shape)
    r = aa.encoder(x_)
    assert r.shape == (x_.shape[0], latent)
    print('enc =', r.shape)

    x_ = torch.randn((7, 400, 24, 24))
    cnn = AATCNN(aa.encoder)
    r = cnn(x_)
    assert r.shape == (x_.shape[0], 1)
    print(x_.shape, r.shape)
