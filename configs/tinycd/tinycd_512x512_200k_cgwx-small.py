_base_ = [
    '../_base_/models/tinycd.py', 
    '../common/train_small_512x512_100k_cgwx.py']

crop_size = (512, 512)
model = dict(
    decode_head=dict(num_classes=2, out_channels=1),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2))
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00356799066427741,
    betas=(0.9, 0.999),
    weight_decay=0.009449677083344786)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)

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