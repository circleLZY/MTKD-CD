_base_ = ['./distill-ban_vit-b16-clip_mit-b0_512x512_200k_cgwx.py']

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
    
    type='DistillBAN',
    encoder_resolution=dict(
        size=(336, 336),
        mode='bilinear'),
    image_encoder=dict(
        type='mmseg.VisionTransformer',
        img_size=(336, 336),
        patch_size=14,
        patch_pad=0,
        embed_dims=1024,
        num_layers=18,
        num_heads=16,
        out_indices=(5, 11, 17)),
    decode_head=dict(
        type='BitemporalAdapterHead',
        ban_cfg=dict(
            fusion_index=[1, 2, 3],
            clip_channels=1024),
        ban_dec_cfg=dict(
            in_channels=[32, 64, 160, 256])))

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=1)
