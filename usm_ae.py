import torch
from wenet.transformer.embedding import RopePositionalEncoding
from wenet.transformer.subsampling import Conv1dSubsampling2
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES
from wenet.utils.mask import make_non_pad_mask

from conformer import ConformerDecoder, ConformerEncoder


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def google_apply_rotary_emb(x: torch.Tensor,
                            freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


class Encoder(torch.nn.Module):

    def __init__(self, configs) -> None:
        super().__init__()
        # TODO: futuer funnel pooling

        self.config = configs
        self.conv = torch.nn.Sequential(
            torch.nn.LayerNorm(configs.input_dim,
                               elementwise_affine=False,
                               bias=False),
            torch.nn.Conv2d(1, configs.input_dim, 3, 2),
            torch.nn.GELU(),
        )
        self.proj_in = torch.nn.Sequential(
            torch.nn.Linear(configs.input_dim,
                            configs.input_dim * 2,
                            bias=False),
            torch.nn.Linear(configs.output_size,
                            configs.input_dim * 2,
                            bias=False))
        self.encoder = ConformerEncoder(configs.encoder)

        self.proj_1 = torch.nn.Linear(configs.output_size,
                                      configs.latent_dim,
                                      bias=False)

        self.proj_2 = torch.nn.Linear(configs.latent_dim,
                                      configs.output_size,
                                      bias=False)
        self.decoder = ConformerDecoder(configs.decoder)

        self.proj_out = torch.nn.Sequential(
            torch.nn.Linear(configs.output_size,
                            configs.input_dim * 2,
                            bias=False),
            torch.nn.Linear(configs.input_dim * 2,
                            configs.input_dim,
                            bias=False))

        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(configs.input_dim,
                                     1,
                                     kernel_size=3,
                                     stride=2,
                                     output_padding=1), torch.nn.ReLU())

    def _encode(self, x: torch.Tensor, x_lens: torch.Tensor):
        x_mask = make_non_pad_mask(x_lens)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.proj_1(x)
        x_mask = x_mask[:, 2::2]  # NOTE may be lost
        x_lens = x_mask.sum(-1)
        hidden, mask = self.encoder(x, x_lens)
        mask = mask.squeeze(1)

        hidden = self.proj_in(hidden)
        return hidden, mask

    def _decode(self, hidden: torch.Tensor, mask: torch.Tensor):
        hidden = self.proj_2(hidden)
        hidden, mask = self.decoder(hidden, mask.sum(-1))
        mask = mask.squeeze(1).float()

        x = self.proj_out(hidden)
        x = self.deconv(x)
        mask = torch.max(mask.unfold(-1, 3, 2), -1).values
        return x, mask

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """forward for training"""

        frame_num = torch.ceil(x.shape[-1] / self.config.input_dim)
        pad_len = frame_num * self.config.input_dim - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad_len), "constant",
                                    0.0).reshape(x.shape[0], -1,
                                                 self.config.input_dim)
        x_lens = x_lens + pad_len

        hidden, mask = self._encode(x, x_lens)
        # TODO: ae
        hidden = self.proj_2(hidden)
        x, x_mask = self._decode(hidden, mask)

        x = x.flatten(-1)
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1,
                                             self.config.input_dim).flatten(-1)
        return x, x_mask.float().sum(-1)
