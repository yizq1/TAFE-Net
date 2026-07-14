_base_ = [
    '../_base_/datasets/rtm_crop.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
num_classes = 2
quality = 80

checkpoint = 'mit_b2_20220624-66e8bf70.pth'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    save_result=dict(type='SegResultHook', interval=1, binary=True))

data_preprocessor = dict(
    type='SegDataPreProcessorWithExtra',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    copy_img = True,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='MyModelFull',
    merge_input=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='AsymCMNeXt_0524',
        use_rectifier=False,
        in_stages=(0,0),
        extra_patch_embed=dict(
            in_channels=64,
            embed_dims=128,
            kernel_size=3,
            stride=1,
            reshape=True,
        ),
        out_indices=(0, 1, 2, 3),

        backbone_main=dict(
            type='MixVisionTransformer',
            pretrained=checkpoint,
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),

        backbone_extra=dict(
            type='HubVisionTransformer0521',
            pretrained='mit_b2_20220624-66e8bf70.pth',
            in_channels=3,
            embed_dims=64,
            modals=['img','img1'],
            in_modals=(2,2,2,2),
            skip_patch_embed_stage=1,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),

        backbone_extra1=dict(
            type='ConvNeXt_0521',
            init_cfg=dict(type='Pretrained', checkpoint='convnextv2_tiny_22k_384_ema_new.pth'),
            in_channels=16,
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.1,
            layer_scale_init_value=0.,
            gap_before_final_norm=False,
            use_grn=True,
            ),

    ),
    preprocessor_sec=[

        [
            'img',
            dict(
                type='LowDctFrequencyExtractor',
            ),
        ],
        [
            'img',
            dict(
                type='HighDctFrequencyExtractor',
            )
        ],

        [
            'dct',
            dict(
                type='Doctamperdct',
            )
        ],
    ],

    neck=dict(
        type='DWTFPN_dct_v6',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),

    decode_head=dict(
        type='SegformerHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=False, reduction='none'),
        ],

    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='ToGray', p=0.5),
    dict(type='OneOf', transforms=[
        dict(type='VerticalFlip', p=0.5),
        dict(type='RandomRotate90', p=0.5),
        dict(type='Transpose', p=0.5),
    ], p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip_withextra', prob=0.5),
    dict(type='RandomFlip_withextra', prob=0.5, direction='vertical'),

    dict(type='RandomCropWithExtra',
         crop_size=crop_size,
         stride=8,
         extra_keys=('img1'),
         cat_max_ratio=0.95),
    dict(type='RandomJpegCompressAndLoadInfo', load_info=True, return_rgb=True, ),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),

    dict(type='RandomJpegCompressAndLoadInfo', load_info=True, return_rgb=True, ),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct'))
]

train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline), num_workers=32)
val_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
))
test_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),

    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        })

        )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=5000)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=800),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=800,
        end=60000,
        by_epoch=False,
    )
]

val_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore', 'aFscore'])
test_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore'])

find_unused_parameters=True
work_dir = './work_dirs/tafenet_segformer_rtm'
