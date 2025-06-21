from typing import Any

import torch
import torchaudio
from absl import app, flags
from exps.conformer_ae.dataset import pad_waveform_to_divisor
from exps.conformer_ae.usm_ae_raw import AudioAutoEncoder
from ml_collections import config_flags

flags.DEFINE_string('wav', None, help='audio file', required=True)
flags.DEFINE_string('checkpoint', None, help='model checkpoint', required=True)

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')


class Pipeline(object):

    def __init__(self, model: AudioAutoEncoder, configs: Any) -> None:
        self.model = model
        self.configs = configs

    def __call__(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
    ):
        xs, xs_mask = self.model._encode(audio, audio_lens)
        mean, _ = xs.chunk(2, dim=-1)

        g = torch.Generator(device=audio.device)
        g.manual_seed(2025)
        noise = torch.randn(audio.shape, generator=g, device=audio.device)
        timesteps = torch.linspace(1,
                                   0,
                                   self.configs.flow_infer_steps + 1,
                                   device=audio.device)
        c_ts = timesteps[:-1]
        p_ts = timesteps[1:]

        latents = noise
        for step in range(self.configs.flow_infer_steps):
            t_curr = c_ts[step]
            t_prev = p_ts[step]
            t_vec = torch.full((noise.shape[0], ),
                               t_curr,
                               dtype=noise.dtype,
                               device=audio.device)
            pred, _ = self.model._decode(latents, xs_mask, mean, t_vec)
            latents = latents + (t_prev - t_curr) * pred

        return latents, xs_mask


def main(_):
    config = FLAGS.config
    print(config)

    # TODO model.from_pretrained
    model = AudioAutoEncoder(config)
    ckpt = torch.load(FLAGS.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    model.cuda()

    pipeline = Pipeline(model, config)
    y, sr = torchaudio.load(FLAGS.wav)
    if sr != config.sample_rate:
        y = torchaudio.functional.resample(y,
                                           orig_freq=sr,
                                           new_freq=config.sample_rate)
    y = pad_waveform_to_divisor(y,
                                config.in_dims).reshape(1, -1,
                                                        config.in_dims).cuda()
    y_len = torch.tensor([y.shape[1]], dtype=torch.int64).cuda()

    outputs = pipeline(y, y_len)
    print(outputs[0].shape)


if __name__ == '__main__':
    app.run(main)
