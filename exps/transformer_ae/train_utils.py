import math
import os
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim as optim
from absl import logging
from exps.transformer_ae.dataset import init_dataset_and_dataloader
from exps.transformer_ae.model import AudioAutoEncoder, kl_divergence
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from vocos.experimental.loss import (MultiScaleSFTLoss,
                                     compute_discriminator_loss,
                                     compute_feature_matching_loss,
                                     compute_generator_loss)
from vocos.experimental.stft_discriminator import MultiScaleSTFTDiscriminator
from wenet.utils.mask import make_non_pad_mask


class TrainModel(torch.nn.Module):

    def __init__(self, model: AudioAutoEncoder, config) -> None:
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor):
        """
        Args:
            audio: shape [B,C,T]
            audio_lens: shape [B,T]
        Returns:
            audio_gen: shape [B,C,T]
            audio_mask: shape [B,T]
            autioencoder mean: shape [B,T//self.config_in_dims, self.config.latent_dim]
            autioencoder logvar: shape [B,T//self.config_in_dims, self.config.latent_dim]
        """
        # NOTE: audio is already padded
        assert audio.ndim == 3 and audio.shape[1] == 1

        B, C, T = audio.shape
        # TODO: conv for multiple channel in future
        audio = audio.reshape(B, C, T // self.config.in_dims, -1).squeeze(1)
        audio_lens = audio_lens // self.config.in_dims

        out, out_mask, loss_kl = self.model(audio, audio_lens)

        # out : [B,T,D] -> [B,1,T*D]
        out = out.reshape(B, 1, T * self.config.in_dims)
        out_mask = out_mask[:, :, None].repeat(1, 1, self.config.in_dims)
        out_mask = out_mask.reshape(B, T * self.config.in_dims)
        return out, out_mask, loss_kl


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps=25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        warmup_steps = self.warmup_steps
        if not isinstance(warmup_steps, List):
            warmup_steps = [self.warmup_steps] * len(self.base_lrs)

        def initlr_fn(lr):
            return lr * step_num**-0.5

        def warmuplr_fn(lr, warmup_step):
            return lr * warmup_step**0.5 * min(step_num**-0.5,
                                               step_num * warmup_step**-1.5)

        return [
            initlr_fn(lr) if warmup_steps[i] == 0 else warmuplr_fn(
                lr, warmup_steps[i]) for (i, lr) in enumerate(self.base_lrs)
        ]

    def set_step(self, step: int):
        self.last_epoch = step


def init_distributed(configs):

    local_rank = os.environ.get('LOCAL_RANK', 0)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    print('training on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))

    return world_size, local_rank, rank


class TrainState:

    def __init__(
        self,
        config: Any,
    ):

        _, _, self.rank = init_distributed(config)
        model = TrainModel(AudioAutoEncoder(config), config).cuda()
        self.config = config
        # TODO: FSDP or deepspeed for future usm ae or dit (flow) decoder
        self.model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        self.mstft_disc = torch.nn.parallel.DistributedDataParallel(
            MultiScaleSTFTDiscriminator(
                config.mstft_disc.filters,
                config.mstft_disc.in_channels,
                config.mstft_disc.out_channels,
                config.mstft_disc.n_ffts,
                config.mstft_disc.hop_lengths,
                config.mstft_disc.win_lengths,
            ).cuda())
        self.spec_loss = MultiScaleSFTLoss(
            config.mstft.n_ffts,
            "same",
            config.mstft.hop_lengths,
            config.mstft.spectralconv_weight,
            config.mstft.log_weight,
            config.mstft.lin_weight,
            config.mstft.phase_weight,
        ).cuda()
        self.device = config.device

        self.max_steps = config.max_train_steps
        _, self.dataloader = init_dataset_and_dataloader(
            config.train_data,
            config.per_device_batch_size,
            config.num_workers,
            config.prefetch,
            True,
            self.max_steps,
            wav_reshape_dims=config.in_dims,
            sample_rate=config.sample_rate,
            seed=config.seed)

        self.step = 0
        self.writer = SummaryWriter(config.tensorboard_dir)
        # Optimizers
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            # betas=(0.8, 0.9),
        )

        self.scheduler = WarmupLR(self.opt, config.warmup_steps)

        # Optimizers
        self.opt_disc = optim.AdamW(
            list(self.mstft_disc.parameters()),
            lr=config.learning_rate,
            betas=(0.8, 0.9),
        )
        # Schedulers
        self.scheduler_disc = WarmupLR(self.opt_disc, config.warmup_steps)

    def train_step(self, batch, device):
        wav, wav_lens = batch['wavs'].to(device), batch['wavs_lens'].to(device)
        log_str = f'[RANK {self.rank}] step_{self.step+1}: '

        wav_g, wavg_mask, loss_kl = self.model(wav, wav_lens)
        wav = wav[:, :, :wav_g.shape[1]]
        wav = wav * wavg_mask[:, None, :]
        if self.config.disc_train_start < self.step + 1:
            self.opt_disc.zero_grad()
            real_score_mrd, real_score_mp_masks, _, _ = self.mstft_disc(
                wav, wavg_mask)
            gen_score_mrd, _, _, _ = self.mstft_disc(wav_g.detach(), wavg_mask)

            loss_mrd, _, _ = compute_discriminator_loss(
                real_score_mrd, gen_score_mrd, real_score_mp_masks)
            disc_loss = loss_mrd

            disc_loss.backward()
            grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
                self.mstft_disc.parameters(), self.config.clip_grad_norm)

            self.opt_disc.step()
            self.scheduler_disc.step()
            if self.rank == 0:
                self.writer.add_scalar("discriminator/multi_res_loss",
                                       loss_mrd, self.step)
                self.writer.add_scalar("discriminator/multi_res_grad_norm",
                                       grad_norm_mrd, self.step)

            log_str += f'loss_mpd: {loss_mrd:>6.3f}'
        spec_loss = self.spec_loss(wav_g, wav, wavg_mask)
        gen_loss = spec_loss * self.config.spec_loss_coeff + self.config.kl_weight * loss_kl

        if self.config.disc_train_start < self.step + 1:
            with torch.no_grad():
                real_score_mrd, _, fmap_rs_mrd, _ = self.mstft_disc(
                    wav, wavg_mask)
            gen_score_mrd, gen_score_mrd_mask, fmap_gs_mrd, fmap_gs_mrd_mask = self.mstft_disc(
                wav_g, wavg_mask)

            loss_gen_mrd, _ = compute_generator_loss(gen_score_mrd,
                                                     gen_score_mrd_mask)
            loss_fm_mrd = compute_feature_matching_loss(
                fmap_rs_mrd, fmap_gs_mrd, fmap_gs_mrd_mask)
            gen_loss = (gen_loss + self.config.mrd_loss_coeff * loss_gen_mrd +
                        loss_fm_mrd + self.config.mrd_loss_coeff * loss_fm_mrd)

        gen_loss.backward()
        grad_norm_g = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.clip_grad_norm)
        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()
        if self.rank == 0:
            if self.config.disc_train_start < self.step + 1:
                self.writer.add_scalar("generator/multi_res_loss",
                                       loss_gen_mrd, self.step)
                self.writer.add_scalar("generator/feature_matching_mrd",
                                       loss_fm_mrd, self.step)
            self.writer.add_scalar("generator/total_loss", gen_loss, self.step)
            self.writer.add_scalar("generator/spec_loss", spec_loss, self.step)
            self.writer.add_scalar("generator/kl_loss", loss_kl, self.step)
            self.writer.add_scalar("generator/grad_norm", grad_norm_g,
                                   self.step)

        log_str += f' loss_gen {gen_loss:>6.3f} mel_loss {spec_loss:>6.3f}'
        if self.config.disc_train_start < self.step + 1:
            opt_disc_lrs = [
                group['lr'] for group in self.opt_disc.param_groups
            ]
            for i, lr in enumerate(opt_disc_lrs):
                if self.rank == 0:
                    self.writer.add_scalar('train/lr_disc_{}'.format(i), lr,
                                           self.step)
                log_str += f' lr_disc_{i} {lr:>6.5f}'
        opt_gen_lrs = [group['lr'] for group in self.opt.param_groups]
        for i, lr in enumerate(opt_gen_lrs):
            if self.rank == 0:
                self.writer.add_scalar('train/lr_gen_{}'.format(i), lr,
                                       self.step)
            log_str += f' lr_gen_{i} {lr:>6.5f}'

        if self.config.decay_mel_coeff:
            self.config.spec_loss_coeff = self.config.base_spec_coeff * max(
                0.0, 0.5 *
                (1.0 + math.cos(math.pi * ((self.step + 1) / self.max_steps))))
        if (self.step + 1) % self.config.log_interval == 0:
            logging.info(log_str)

    def train(self):
        if self.config.checkpoint != '':
            self.resume(self.config.checkpoint)
        self.model.train()
        for batch in self.dataloader:
            dist.barrier()
            self.train_step(batch, self.config.device)
            if (self.step + 1) % self.config.checkpoint_every_steps == 0:
                self.save()
            self.step += 1
            if self.step >= self.max_steps:
                print("Training complete.")
                return

    def save(self):
        if self.rank == 0:
            checkpoint_dir = os.path.join(self.config.model_dir,
                                          f'step_{self.step}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            model_state_dict = self.model.module.state_dict()
            meta = {
                'model': model_state_dict,
                'step': self.step,
            }
            torch.save(meta, os.path.join(checkpoint_dir, 'model.pt'))

            disc_state_dict = self.model.module.state_dict()
            torch.save(disc_state_dict, os.path.join(checkpoint_dir,
                                                     'disc.pt'))

            opt_dict = self.opt.state_dict()
            torch.save(opt_dict, os.path.join(checkpoint_dir, 'opt.pt'))

            opt_dict = self.opt_disc.state_dict()
            torch.save(opt_dict, os.path.join(checkpoint_dir, 'opt_disc.pt'))

            scheduler_state_dict = self.scheduler.state_dict()
            torch.save(scheduler_state_dict,
                       os.path.join(checkpoint_dir, 'scheduler.pt'))

            scheduler_state_dict = self.scheduler_disc.state_dict()
            torch.save(scheduler_state_dict,
                       os.path.join(checkpoint_dir, 'disc_scheduler.pt'))

            logging.info(
                f'[RANK {self.rank}] Checkpoint: save to checkpoint {checkpoint_dir}'
            )

    def resume(self, checkpoint_dir: str):

        model = self.model.module
        ckpt = torch.load(os.path.join(checkpoint_dir, 'model.pt'),
                          map_location='cpu',
                          mmap=True)
        model.load_state_dict(ckpt['model'])
        self.step = ckpt['step'] + 1  # train from new step

        disc = self.mstft_disc.module
        ckpt = torch.load(os.path.join(checkpoint_dir, 'disc.pt'),
                          map_location='cpu',
                          mmap=True)
        disc.load_state_dict(ckpt)

        opt = self.opt
        ckpt = torch.load(os.path.join(checkpoint_dir, 'opt.pt'),
                          map_location='cpu',
                          mmap=True)
        opt.load_state_dict(ckpt)

        opt = self.opt_disc
        ckpt = torch.load(os.path.join(checkpoint_dir, 'opt_disc.pt'),
                          map_location='cpu',
                          mmap=True)
        opt.load_state_dict(ckpt)

        logging.info(
            f'[RANK {self.rank}] Checkpoint: load  checkpoint {checkpoint_dir}'
        )

        dist.barrier()
