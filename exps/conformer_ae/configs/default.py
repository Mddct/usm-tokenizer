import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.model_type = 'conformer'
    config.train_data = ''
    config.eval_data = ''
    config.model_dir = ''
    config.tensorboard_dir = ''
    config.num_workers = 10
    config.prefetch = 100
    config.log_interval = 100
    config.device = 'cuda'
    config.checkpoint = ''

    # train
    # Per device batch size for training.
    config.per_device_batch_size = 32
    # Per device batch size for training.
    config.eval_per_device_batch_size = 32
    config.max_train_steps = 500_000
    config.num_eval_steps = 2_000
    # Base learning rate.
    config.learning_rate = 0.0016
    # Linear learning rate warmup.
    config.warmup_steps = 1000
    # Decay factor for AdamW style weight decay.
    config.weight_decay = 0.1
    # Save a checkpoint every these number of steps.
    config.checkpoint_every_steps = 10_000
    # Frequency of eval during training, e.g. every 1_000 steps.
    config.eval_every_steps = 1_000
    # Use bfloat16 mixed precision training instead of float32.
    config.use_bfloat16 = False
    # Integer for PRNG random seed.
    config.seed = 2025
    config.clip_grad_norm = 1

    # mel
    config.sample_rate = 24000
    config.hop_size = 480  # sample_rate // hop_size = 50 for flow
    config.n_fft = 1920  # hop_size * 4
    config.n_mels = 80  # 128 for future
    config.power = 1
    config.fmin = 0
    config.fmax = None
    config.norm = 'slaney'
    config.mel_scale = 'slaney'
    config.padding = "center"
    config.multiscale_mel_loss = True

    # loss
    config.mel_loss_coeff = 45
    config.mrd_loss_coeff = 1.0  # 0.1 for fintune
    config.pretrain_mel_steps = 0
    config.decay_mel_coeff = False
    config.disc_train_start = 0
    config.mrd = True

    # TODO(Mddct:) other info

    # model

    config.output_size = 256
    config.attention_heads = 4
    config.linear_units = 2048
    config.dropout_rate = 0.1
    config.positional_dropout_rate = 0.1
    config.attention_dropout_rate = 0.0
    config.normalize_before = True
    config.query_bias = True
    config.key_bias = True
    config.value_bias = True
    config.activation_type = "relu"
    config.gradient_checkpointing = False
    config.use_sdpa = False
    config.layer_norm_type = "rms_norm"
    config.norm_eps = 1e-5
    config.n_kv_head = None
    config.head_dim = None
    config.selfattention_layer_type = "selfattn"
    config.mlp_type = "moe"  # ['position_wise_feed_forward', 'moe', 'gated']
    config.mlp_bias = True
    config.n_expert = 8
    config.n_expert_activated = 2
    config.right_context = 2
    config.left_context = 15

    # total blocks: first_n_layers + num_blocks
    config.first_n_layers = 3
    config.causal_blocks = 3
    config.noncausal_blocks = 6
    config.causal = True
    config.cnn_module_kernel = 15
    config.use_cnn_module = True
    config.final_norm = False
    config.cnn_module_norm = "rms_norm"
    config.conv_bias = True
    config.conv_norm_eps = 1e-6
    config.conv_inner_factor = 2
    config.macaron_style = True

    config.latent_dim = 16
    config.in_dims = 320  # 50hz, 16K: 320; 24k 480, 48k 960
    config.flow_infer_steps = 50
    config.loss_kl_weight = 0.000001
    config.timesteps_dim = 32

    return config
