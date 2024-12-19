_base_ = ['./ban_vit-b16-clip_mit-b0_512x512_200k_cgwx.py']

dataset_type = 'LEVIR_CD_Dataset'
data_root = '/nas/datasets/lzy/RS-ChangeDetection/CGWX-Extended'

pretrained = 'pretrain/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth'  # noqa

model = dict(
    type='BAN',
    pretrained=pretrained,
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


test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # for val
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val/label',
            img_path_from='val/A',
            img_path_to='val/B'),
        # data_prefix=dict(
        #     seg_map_path='test/label',
        #     img_path_from='test/A',
        #     img_path_to='test/B'),
        pipeline=test_pipeline))


# optimizer
optimizer=dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=400000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 400k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=400000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, 
                       img_shape=(512, 512, 3)))