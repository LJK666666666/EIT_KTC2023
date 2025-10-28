


from ..third_party_models import FCUNet

def get_fcunet(config):

    model = FCUNet(
        image_size=config.data.im_size,
        in_channels=config.model.in_channels,
        model_channels=config.model.model_channels,
        out_channels=config.model.out_channels,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        channel_mult=config.model.channel_mult,
        conv_resample=config.model.conv_resample,
        dims=config.model.dims,
        num_heads=config.model.num_heads,
        num_head_channels=config.model.num_head_channels,
        num_heads_upsample=config.model.num_heads_upsample,
        use_scale_shift_norm=config.model.use_scale_shift_norm,
        resblock_updown=config.model.resblock_updown,
        use_new_attention_order=config.model.use_new_attention_order,
        max_period=config.model.max_period
        )

         
    return model
