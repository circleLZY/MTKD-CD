_base_ = ['./ban_vit-l14-clip_mit-b0_512x512_200k_cgwx.py']

pretrained = 'pretrain/RS5M_ViT-L-14-336.pth'  # noqa

model = dict(
    pretrained=pretrained)