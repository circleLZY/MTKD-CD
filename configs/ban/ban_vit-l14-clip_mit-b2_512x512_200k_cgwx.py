_base_ = ['./ban_vit-l14-clip_mit-b0_512x512_200k_cgwx.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# model settings
model = dict(
    decode_head=dict(
        ban_cfg=dict(
            side_enc_cfg=dict(
                init_cfg=dict(
                    type='Pretrained', checkpoint=checkpoint),
                embed_dims=64,
                num_layers=[3, 4, 6, 3])),
        ban_dec_cfg=dict(
            in_channels=[64, 128, 320, 512])))

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=1)