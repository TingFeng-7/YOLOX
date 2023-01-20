#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp

config_root = os.path.abspath(os.path.split(__file__)[0])
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33 
        self.width = 0.25
        # 新数据按这个：['text', 'icon', 'image','linebox']
        # 工程段最后需要转成这样
        # 0:bg , 1:icon, 2:text,3:image, 4:line box
        self.cls_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        self.num_classes = len(self.cls_names)
        
        self.data_num_workers = 2
        self.basic_lr_per_img = 0.01 / 4
        self.max_epoch = 200
        self.print_interval = 20
        self.eval_interval = 20

        self.input_size = (720, 1280)
        self.multiscale_range = 0
  
        self.mosaic_prob = 0.0 #关闭所有数据增强
        self.mixup_prob = 0.0 #关闭所有数据增强
        self.degrees = 0.0
        self.shear = 0.0
        self.enable_mixup = False
        self.mosaic_scale = (0.5, 1.5) 

        self.test_size = self.input_size
        
        import datetime
        nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.exp_name = 'outputs_'+nowTime

 
        # Define yourself dataset path
        # self.data_dir = "datasets/coco128"
        data_root = "/root/visDrone2019/images"
        # 在这里找的 
        self.data_dir = data_root
        self.train_ann = '/root/visDrone2019/train.json'
        self.val_ann = '/root/visDrone2019/train.json'
        # 
        # self.in_channels = [128, 256, 512, 1024]
        # self.in_features = ("dark2", "dark3", "dark4", "dark5")
        # self.strides = [4, 8, 16, 32]
        self.in_channels = [256, 512, 1024]
        self.in_features = ("dark3", "dark4", "dark5")
        self.strides = [8, 16, 32]

        # name of LRScheduler 
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 0

        self.output_dir = config_root
        self.scaled32 = False
        # self.eval_interval = 20
        self.save_history_ckpt = False

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, YOLOPAFPN_hbo
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN_hbo(self.depth, self.width, in_channels=self.in_channels,
                                 in_features=self.in_features, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=self.in_channels, depthwise=True,
                strides = self.strides)

            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model


# exps/test07_v1.5_iou1_3000epoch_ddp/config.py 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/test08_v1.6_iou1_3000epoch_ddp_right_index/config.py -b 32 -d 4 --fp16 -o --resume -c exps/test08_v1.6_iou1_3000epoch_ddp_right_index/ou

# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/test08_v1.6_iou1_3000epoch_ddp_right_index/config.py -b 32 -d 4 --fp16 -o -c models/yolox_nano.pth
 
#docker attach  yolox-all1
# CUDA_VISIBLE_DEVICES=1 python tools/train.py -f exps/test09_0010_v1.5_filtered_linebox/config.py -b 8 -d 1 --fp16 -o -c models/yolox_nano.pth
# CUDA_VISIBLE_DEVICES=1 python tools/train.py -f exps/test09_0010_v1.5_filtered_linebox/config.py -b 8 -d 1 --fp16 -o -c exps/test09_0010_v1.5_filtered_linebox/outputs_2022-12-07-09-49-08/best_ckpt.pth --resume

