_base_ = ['./lightcdnet_s_512x512_200k_cgwx.py']

model = dict(
    backbone=dict(net_type="base"),
    neck=dict(in_channels=[24, 116, 232, 464]))