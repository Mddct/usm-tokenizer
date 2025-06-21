# exponential: raw wav as ae encoder, one channel
# 16K: reshape [..., 320]
# 24k: reshape [..., 480]
# 48k: reshape [..., 960] no need to 768

# all 50hz
# TODO: 100hz -> 50hz -> 25hz

import math
from re import match
from typing import Any, Tuple

import torch
from efficient_conformer.model import Conformer
from wenet.utils.mask import make_non_pad_mask

Configs = Any


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(logvar.shape)
    return mean + eps * std


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:

    t = timesteps[:, None, None]
    noisy_samples = t * noise + (1 - t) * original_samples
    return noisy_samples


class TimestepEmbedding(torch.nn.Module):

    def __init__(self, in_size, embedding_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_size, embedding_dim)
        self.linear2 = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        return x


class SimpleDIT(torch.nn.Module):

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        self.decoder = Conformer(configs)

        self.time_embed = TimestepEmbedding(configs.latent_dim,
                                            configs.output_size)

    def timestep_embedding(self,
                           t: torch.Tensor,
                           dim: int,
                           max_period: float = 10000,
                           time_factor: float = 1000.0) -> torch.Tensor:
        """
        Generate sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): shape (batch,), can be float.
            dim (int): embedding dimension.
            max_period (float): controls the minimum frequency of the embeddings.
            time_factor (float): scales the input timestep (default 1000.0)

        Returns:
            torch.Tensor: shape (batch, dim), timestep embeddings.
        """

        t = t * time_factor  # Scale time
        half = dim // 2

        device = t.device
        dtype = t.dtype

        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(0, half, dtype=torch.float32, device=device) /
            half).to(dtype)  # shape: (half,)

        args = t[:, None] * freqs[None, :]  # shape: (batch, half)
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)],
            dim=-1)  # shape: (batch, dim or dim - 1)

        if dim % 2 != 0:
            # Pad with zeros if dim is odd
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(
        self,
        latent: torch.Tensor,
        latent_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """forward for training
        """
        timesteps = self.timestep_embedding(timesteps,
                                            self.configs.latent_dim)  # [B, D]
        temb = self.time_embed(timesteps)

        hidden_states = temb + latent
        v_pred, mask = self.decoder(hidden_states, latent_mask)
        return v_pred, mask.squeeze(1)


class AudioAutoEncoder(torch.nn.Module):

    def __init__(self, configs: Configs) -> None:
        super().__init__()
        self.configs = configs

        self.encoder = Conformer(configs)

        self.decoder = SimpleDIT(configs)
        self.down = torch.nn.Sequential(
            torch.nn.Linear(configs.output_size, 768, bias=False),
            torch.nn.Linear(768, configs.latent_dim * 2, bias=False),
        )
        self.up = torch.nn.Sequential(
            torch.nn.Linear(configs.latent_dim, 768, bias=False),
            torch.nn.Linear(768, configs.output_size, bias=False),
        )

        self.proj_in = torch.nn.Sequential(
            torch.nn.Linear(configs.in_dims, 768, bias=False),
            torch.nn.Linear(768, configs.output_size, bias=False),
        )

        self.proj_out = torch.nn.Linear(configs.output_size, configs.in_dims)
        assert self.configs.final_norm is False

    def _encode(self, audio: torch.Tensor,
                audio_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: audio is padded
        assert audio.shape[-1] in [320, 480, 960]

        xs = self.proj_in(audio)  # [B,T,D]
        xs_mask = make_non_pad_mask(audio_lens)

        encoder_out, _ = self.encoder(xs, xs_mask)
        out = self.down(encoder_out)
        return out, xs_mask.squeeze(1)

    def _decode(self, z: torch.Tensor, z_mask: torch.Tensor, t: torch.Tensor):
        xs = self.up(z)
        xs, _ = self.decoder(xs, z_mask, t)

        return xs, z_mask

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor,
                t: torch.Tensor):
        """ forward for training
        """
        xs, xs_mask = self._encode(audio, audio_lens)
        mean, log_var = xs.chunk(2, dim=-1)
        z = reparameterize(mean, log_var)

        noise = torch.randn(z.shape, dtype=z.dtype, device=z.device)
        noise_z = add_noise(z, noise, t)
        xs, _ = self._decode(noise_z, xs_mask.squeeze(1), t)

        recog = self.proj_out(xs)

        target = noise - z
        loss_flow = (target - recog)**2 * xs_mask.unsqueeze(-1)
        loss_kl = kl_divergence(mean, log_var) * xs_mask.unsqueeze(-1)

        loss_flow = loss_flow.sum() / xs_mask.sum() / self.configs.latent_dim
        loss_kl = loss_kl.sum() / xs_mask.sum() / self.configs.latent_dim
        loss = self.configs.loss_kl_weight * loss_kl.sum() + loss_flow
        return {
            "loss": loss,
            "loss_kl": loss_kl.sum().detach(),
            "loss_flow": loss_flow.sum().detach(),
        }


def kl_divergence(mean, logvar):
    return -0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar))


if __name__ == '__main__':
    from exps.conformer_ae.configs.default import get_config
    from exps.conformer_ae.usm_ae_raw import AudioAutoEncoder

    configs = get_config()
    model = AudioAutoEncoder(configs)

    wav = torch.randn(1, 50, configs.in_dims)
    wav_lens = torch.tensor([50], dtype=torch.long)
    timesteps = torch.linspace(1, 0, steps=configs.flow_infer_steps + 1)

    ts = torch.randint(
        low=0,
        high=len(timesteps) - 1,
        size=(wav.shape[0], ),
    )
    t = timesteps[ts]

    output = model(wav, wav_lens, t)
    print(output)
