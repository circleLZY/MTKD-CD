_base_ = ['./ban_vit-b16-clip_mit-b0_512x512_200k_cgwx.py']

pretrained = 'pretrain/RemoteCLIP-ViT-B-32.pth'  # noqa

model = dict(
    pretrained=pretrained,
    image_encoder=dict(patch_size=32))