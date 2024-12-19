_base_ = [
    '../_base_/models/cgnet.py', 
    '../common/train_large_512x512_100k_cgwx.py']

dataset_type = 'LEVIR_CD_Dataset'
data_root = '/nas/datasets/lzy/RS-ChangeDetection/CGWX'

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0025)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)

crop_size = (512, 512)


model = dict(
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=200000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 100k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=200000, val_interval=2500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2500,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, 
                       img_shape=(512, 512, 3)))