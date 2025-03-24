# 在 custom_backbones 目录下新建 hiera_backbone.py
from mmengine.model import BaseModule
from timm.models.hiera import hiera
import torch.nn as nn
from mmdet.registry import MODELS
import timm


@MODELS.register_module()
class HieraBackbone(BaseModule):
    def __init__(self, 
                 model_name='hiera_tiny_224.mae',
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 pretrained=True):
        super().__init__()
        
        self.timm_model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_indices=out_indices
        # 冻结参数设置
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # 冻结 stem 和所有 stage
            for param in self.timm_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.timm_model(x)
        return tuple([features[i] for i in self.out_indices])

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()  # 保证冻结状态在训练模式切换时维持
