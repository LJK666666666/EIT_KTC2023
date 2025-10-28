import ml_collections

def get_default_configs():

    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1
    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type =  'ddpm'

    sde.beta_min = 0.0001
    sde.beta_max = 0.02
    sde.num_steps = 1000


    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 12
    training.epochs = 1000
    training.log_freq = 20
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = 50 # only start updating ema after this amount of steps 
    training.save_model_every_n_epoch = 10

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()

    validation.num_steps = 100
    validation.sample_freq = 1 # 0 = NO VALIDATION SAMPLES DURING TRAINING
    validation.eps = 1e-3

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.eps = 1e-3
    sampling.travel_length = 1
    sampling.travel_repeat = 1
    
    
    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    model.in_channels = 6
    model.model_channels = 64
    model.out_channels = 1
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
    model.max_period = 1e4
   
    # data configs - specify in other configs
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256

    config.sampling.load_model_from_path = None 
    config.sampling.model_name = None 

    return config