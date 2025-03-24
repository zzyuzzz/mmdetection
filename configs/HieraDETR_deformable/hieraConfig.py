_base_ = ['../deformable_detr/deformable-detr_r50_16xb2-50e_coco.py',
            # '../_base_/datasets/coco_detection.py',
            # '../_base_/default_runtime.py'
            ]

# custom_imports = dict(
#     imports=['mmdet.models.backbones.hiera'],
#     allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='HieraBackbone',
        model_name='hiera_base_224.mae',
        pretrained=True,
        frozen_stages=4,  # 冻结全部参数
        out_indices=(1, 2, 3,)
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1)
)

load_from = r'.\work_dirs\hieraConfig\20250224_224303\epoch_2.pth'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline),batch_size=64)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 15
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

auto_scale_lr = dict(enable=True, base_batch_size=64)