from typing import Tuple

import torch
from torch import nn

# Added constant for solar radius (meters)
RSUN_METERS = 4*6.9634e8


class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 3,
            d_output: int = 1,
            n_layers: int = 8,
            d_filter: int = 512,
            skip: Tuple[int] = (),
            encoding='positional',
            device: str = 'cuda:0',
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = Sine()

        # encoding_config = {'type': 'positional', 'num_freqs': 20} if encoding_config is None else encoding_config
        # encoding_type = encoding_config.pop('type')
        if encoding == 'positional':
            enc = PositionalEncoding(d_input=d_input, n_freqs=13, scale_factor=1.)
            in_layer = nn.Linear(enc.d_output, d_filter)
            self.in_layer = nn.Sequential(enc, in_layer)
        else:
            self.in_layer = nn.Linear(d_input, d_filter)
        # elif encoding_type == 'none' or encoding_type is None:
        #     self.in_layer = nn.Linear(d_input, d_filter)
        # else:
        #     raise ValueError(f'Unknown encoding type {encoding_type}')

        # Create model layers
        self.layers = nn.ModuleList([nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)])

        self.out_layer = nn.Linear(d_filter, d_output)
        
        # Move model to specified device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        Now normalizes input coordinates by solar radius to keep values ~O(1).
        """

        # Normalize coordinates by solar radius
        rsun = torch.as_tensor(RSUN_METERS, dtype=x.dtype, device=x.device)
        x = x / rsun

        # Debug prints (kept)
        # print("Input to NeRF (normalized by R_sun):", x)
        # print("Input device:", x.device)
        # print("in_layer device:", next(self.in_layer.parameters()).device)
        # for i, layer in enumerate(self.layers):
        #     print(f"Layer {i} device:", next(layer.parameters()).device)
        # print("out_layer device:", next(self.out_layer.parameters()).device)
        x = self.act(self.in_layer(x))
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            # if i in self.skip:
            #     x = torch.cat([x, x_input], dim=-1)
        x = self.out_layer(x)

        # print("Output passed through NeRF:", x)

        return x


class EmissionModel(NeRF):

    def __init__(self, device: str = 'cuda:0', **kwargs):
        super().__init__(d_input=4, d_output=2, device=device, **kwargs)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class TrainablePositionalEncoding(nn.Module):

    def __init__(self, d_input, n_freqs=20):
        super().__init__()
        frequencies = torch.stack([torch.linspace(-3, 9, n_freqs, dtype=torch.float32) for _ in range(d_input)], -1)
        self.frequencies = nn.Parameter(frequencies[None, :, :], requires_grad=True)
        self.d_output = n_freqs * 2 * d_input

    def forward(self, x):
        # x = (batch, rays, coords)
        encoded = x[:, None, :] * torch.pi * 2 ** self.frequencies
        normalization = (torch.pi * 2 ** self.frequencies)
        encoded = torch.cat([torch.sin(encoded) / normalization, torch.cos(encoded) / normalization], -1)
        encoded = encoded.reshape(x.shape[0], -1)
        return encoded


class PositionalEncoding(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, scale_factor: float = 2., log_space: bool = True):
        """

        Parameters
        ----------
        d_input: number of input dimensions
        n_freqs: number of frequencies used for encoding
        scale_factor: factor to adjust box size limit of 2pi (default 2; 4pi)
        log_space: use frequencies in powers of 2
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)
            print('freq bands', freq_bands)

        self.register_buffer('freq_bands', freq_bands)
        self.scale_factor = scale_factor

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        f = self.freq_bands[None, :, None]
        enc = [x,
               (torch.sin(x[:, None, :] * f / self.scale_factor)).reshape(x.shape[0], -1),
               (torch.cos(x[:, None, :] * f / self.scale_factor)).reshape(x.shape[0], -1)
               ]
        return torch.concat(enc, dim=-1)
