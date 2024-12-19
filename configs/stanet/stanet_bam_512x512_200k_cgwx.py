_base_ = './stanet_base_512x512_200k_cgwx.py'

crop_size = (512, 512)
model = dict(
    decode_head=dict(sa_mode='BAM'),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )