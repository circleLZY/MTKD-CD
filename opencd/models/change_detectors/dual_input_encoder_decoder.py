# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class DIEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        x = self.backbone(img_from, img_to)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class DistillDIEncoderDecoder_S(DIEncoderDecoder):
    def __init__(self,
                #  teacher_l: DIEncoderDecoder,  # 教师模型
                #  teacher_m: DIEncoderDecoder,  # 教师模型
                #  teacher_s: DIEncoderDecoder,  # 教师模型
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        # self.teacher_l = MODELS.build(teacher_l)
        # self.teacher_m = MODELS.build(teacher_m)
        # self.teacher_s = MODELS.build(teacher_s)
        self.teacher_l = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_m = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_m,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_s = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
        # 确保教师模型不参与参数更新
        # for param in self.teacher_l.parameters():
        #     param.requires_grad = False
        # for param in self.teacher_m.parameters():
        #     param.requires_grad = False
        # for param in self.teacher_s.parameters():
        #     param.requires_grad = False
            
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward(self.teacher_m.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses


@MODELS.register_module()
class DistillDIEncoderDecoder_S_TwoTeachers(DIEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        self.teacher_l = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )

        self.teacher_s = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
            
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.10:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses
