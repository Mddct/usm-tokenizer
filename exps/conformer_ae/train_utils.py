import os
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim as optim
from absl import logging
from exps.conformer_ae.dataset import init_dataset_and_dataloader
from exps.conformer_ae.usm_ae_raw import AudioAutoEncoder
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter


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
        model = AudioAutoEncoder(config)
        model.cuda()
        self.config = config
        # TODO: FSDP or deepspeed for future usm ae or dit (flow) decoder
        self.model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
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

    def train_step(self, batch, device):
        wav, wav_lens = batch['audio'].to(device), batch['audio_lens'].to(
            device)

        log_str = f'[RANK {self.rank}] step_{self.step+1}: '

        loss_dict = self.model(wav, wav_lens,
                               torch.rand(wav.shape[0], device=wav.device))
        loss = loss_dict['loss']
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.config.clip_grad_norm)
        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()
        if self.rank == 0:
            self.writer.add_scalar("train/loss_flow", loss_dict['loss_flow'],
                                   self.step)
            self.writer.add_scalar("train/loss_kl", loss_dict['loss_flow'],
                                   self.step)
            self.writer.add_scalar("train/loss", loss_dict['loss'], self.step)

            self.writer.add_scalar("train/grad_norm", grad_norm, self.step)

        log_str += f' loss {loss:>6.3f} loss_flow {loss_dict["loss_flow"]:>6.3f} loss_kl {loss_dict["loss_kl"]:>6.3f}'
        opt_lrs = [group['lr'] for group in self.opt.param_groups]
        for i, lr in enumerate(opt_lrs):
            if self.rank == 0:
                self.writer.add_scalar('train/lr_{}'.format(i), lr, self.step)
            log_str += f' lr_{i} {lr:>6.5f}'

        log_str += f" grad_norm {grad_norm:>6.3f}"
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

            opt_dict = self.opt.state_dict()
            torch.save(opt_dict, os.path.join(checkpoint_dir, 'opt.pt'))

            scheduler_state_dict = self.scheduler.state_dict()
            torch.save(scheduler_state_dict,
                       os.path.join(checkpoint_dir, 'scheduler.pt'))
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

        opt = self.opt
        ckpt = torch.load(os.path.join(checkpoint_dir, 'opt_disc.pt'),
                          map_location='cpu',
                          mmap=True)
        opt.load_state_dict(ckpt)

        logging.info(
            f'[RANK {self.rank}] Checkpoint: load  checkpoint {checkpoint_dir}'
        )

        dist.barrier()
