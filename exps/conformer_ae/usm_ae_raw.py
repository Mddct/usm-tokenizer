# exponential: raw wav as ae encoder, one channel
# 16K: reshape [..., 320]
# 24k: reshape [..., 480]
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


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:

    t = timesteps[:, None, None]
    noisy_samples = (1 - t) * noise + t * original_samples
    return noisy_samples


class SimpleDIT(torch.nn.Module):

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        self.decoder = Conformer(configs)
        # TODO: layer by layer time constraint
        self.t_embedding = torch.nn.Sequential(
            torch.nn.Linear(configs.timesteps_dim, configs.timesteps_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(configs.timesteps_dim * 2, configs.output_size),
        )
        self.proj = torch.nn.Linear(configs.in_dims + configs.output_size,
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
        xs: torch.Tensor,
        xs_mask: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """forward for training
        """
        timesteps = self.timestep_embedding(
            timesteps, self.configs.timesteps_dim)  # [B, D]
        t_emb = self.t_embedding(timesteps)
        c_emb = condition

        # TODO: cfg
        xs = torch.cat((xs, c_emb), dim=-1)
        hidden_states = self.proj(xs)
        hidden_states = t_emb.unsqueeze(1) + hidden_states
        v_pred, mask = self.decoder(hidden_states, xs_mask)
        return v_pred, mask.squeeze(1)


class AudioAutoEncoder(torch.nn.Module):

    def __init__(self, configs: Configs) -> None:
        super().__init__()
        assert configs.final_norm is False

        self.configs = configs

        self.encoder = Conformer(configs)
        self.z_proj = torch.nn.Linear(configs.output_size,
                                      configs.output_size * 2,
                                      bias=False)
        with torch.no_grad():
            for param in self.encoder.parameters():
                param *= 0.5

        self.decoder = SimpleDIT(configs)
        self.enc_proj_in = torch.nn.Sequential(
            torch.nn.Linear(configs.in_dims, 768, bias=False),
            torch.nn.Linear(768, configs.output_size),
        )
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

    def _decode(self, z: torch.Tensor, z_mask: torch.Tensor,
                condition: torch.Tensor, t: torch.Tensor):
        xs, _ = self.decoder(z, z_mask, condition, t)
        xs = self.dec_proj_out(xs)
        return xs, z_mask

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor,
                t: torch.Tensor):
        """ forward for training
        """
        xs, xs_mask = self._encode(audio, audio_lens)
        mean, log_var = xs.chunk(2, dim=-1)
        z = reparameterize(mean, log_var)
        condition = z

        noise = torch.randn(audio.shape,
                            dtype=audio.dtype,
                            device=audio.device)
        noise_audio = add_noise(audio, noise, t)
        recog, _ = self._decode(noise_audio, xs_mask, condition, t)

        target = audio - noise
        loss_flow = (target - recog)**2 * xs_mask.unsqueeze(-1)
        loss_kl = kl_divergence(mean, log_var) * xs_mask.unsqueeze(-1)

        loss_flow = loss_flow.sum() / (xs_mask.sum() * self.configs.in_dims)
        loss_kl = loss_kl.sum() / xs_mask.sum()
        loss = self.configs.loss_kl_weight * loss_kl.sum() + loss_flow
        return {
            "loss": loss,
            "loss_kl": loss_kl.detach(),
            "loss_flow": loss_flow.detach(),
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
