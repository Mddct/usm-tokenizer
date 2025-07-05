# exponential: raw wav as ae encoder, one channel
# 48k: reshape [..., 960] no need to 768

# all 50hz
# TODO: 100hz -> 50hz -> 25hz

import math
from typing import Any, Tuple

import torch
from efficient_conformer.model import Conformer
from wenet.utils.mask import make_non_pad_mask

Configs = Any


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(logvar.shape, device=mean.device)
    return mean + eps * std


class AudioAutoEncoder(torch.nn.Module):

    def __init__(self, configs: Configs) -> None:
        super().__init__()
        assert configs.final_norm is False

        self.configs = configs

        self.encoder = Conformer(configs)
        self.z_proj = torch.nn.Linear(configs.output_size,
                                      configs.latent_dim * 2,
                                      bias=False)
        self.z_proj_out = torch.nn.Linear(configs.latent_dim,
                                          configs.output_size,
                                          bias=Fasle)
        with torch.no_grad():
            for param in self.encoder.parameters():
                param *= 0.5

        self.decoder = Conformer(configs)
        self.enc_proj_in = torch.nn.Linear(configs.in_dims,
                                           configs.output_size,
                                           bias=False)
        self.dec_proj_out = torch.nn.Linear(configs.output_size,
                                            configs.in_dims)

    def _encode(self, audio: torch.Tensor,
                audio_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: audio is padded
        assert audio.shape[-1] in [320, 480, 960]

        xs = self.enc_proj_in(audio)  # [B,T,D]
        xs_mask = make_non_pad_mask(audio_lens)

        encoder_out, _ = self.encoder(xs, xs_mask)
        z = self.z_proj(encoder_out)
        return z, xs_mask

    def _decode(self, inputs: torch.Tensor, inputs_mask: torch.Tensor):
        xs = self.z_proj_out(inputs)
        xs, _ = self.decoder(xs, inputs_mask)
        xs = self.dec_proj_out(xs)
        return xs, inputs_mask

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor):
        """ forward for training
        """
        xs, xs_mask = self._encode(audio, audio_lens)
        mean, log_var = xs.chunk(2, dim=-1)
        z = reparameterize(mean, log_var)

        out, _ = self._decode(x, xs_mask)
        out_mask = xs_mask
        return out, out_mask
