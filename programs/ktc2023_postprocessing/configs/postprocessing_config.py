import ml_collections

def get_configs():

    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1
    # sde configs


    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 6
    training.epochs = 1000
    training.log_freq = 20
    training.lr = 1e-4
    training.save_model_every_n_epoch = 10


    
    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    model.in_channels = 5
    model.model_channels = 64
    model.out_channels = 3
    model.num_res_blocks = 2
    model.attention_resolutions = [16, 32]
    model.channel_mult = (1., 1., 2., 2., 4., 4.) 
    model.conv_resample = True
    model.dims = 2
    model.num_heads = 2
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.use_scale_shift_norm = True 
    model.resblock_updown = False
    model.use_new_attention_order = False
    model.max_period = 100
   
    # data configs - specify in other configs
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256

    return config