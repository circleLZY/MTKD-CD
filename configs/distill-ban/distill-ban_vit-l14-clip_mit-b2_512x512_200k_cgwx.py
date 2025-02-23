_base_ = ['./distill-ban_vit-l14-clip_mit-b0_512x512_200k_cgwx.py']

checkpoint_student = None
checkpoint_teacher_l = None
checkpoint_teacher_m = None
checkpoint_teacher_s = None

# model settings
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher medium    
    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),
    
    decode_head=dict(
        ban_cfg=dict(
            side_enc_cfg=dict(
                embed_dims=64,
                num_layers=[3, 4, 6, 3])),
        ban_dec_cfg=dict(
            in_channels=[64, 128, 320, 512])))

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=1)