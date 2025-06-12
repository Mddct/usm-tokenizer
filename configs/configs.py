import ml_collections


def _config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.output_size = 1024
    config.attention_heads = 8
    config.linear_units = 2048
    config.dropout_rate = 0.1
    config.positional_dropout_rate = 0.1
    config.attention_dropout_rate = 0.0
    config.normalize_before = True
    config.query_bias = False
    config.key_bias = False
    config.value_bias = False
    config.activation_type = "relu"
    config.gradient_checkpointing = False
    config.use_sdpa = False
    config.layer_norm_type = "rms_norm"
    config.norm_eps = 1e-5
    config.n_kv_head = None
    config.head_dim = 128
    config.selfattention_layer_type = "selfattn"
    config.mlp_type = "moe"  # ['position_wise_feed_forward', 'moe', 'gated']
    config.mlp_bias = False
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
    config.conv_bias = False
    config.conv_norm_eps = 1e-6
    config.conv_inner_factor = 2
    config.macaron_style = True

    return config


def get_config():
    config = ml_collections.ConfigDict()
    config.encoder = _config()
    config.decoder = _config()

    config.input_dim = 160  # 16000 / 100 = 160 24000/100 = 240 44000/100 = 440
    config.output_size = config.encoder.output_size

    config.latent_dim = 16

    return config
