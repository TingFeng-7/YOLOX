#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.cls_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        self.num_classes = len(self.cls_names)
        
        self.data_num_workers = 2
        self.basic_lr_per_img = 0.01 / 4
        self.max_epoch = 200
        self.print_interval = 20
        self.eval_interval = 20

        self.input_size = (736, 1280)
        self.multiscale_range = 0
        self.test_size = self.input_size
    
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        
        import datetime
        nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config_root = os.path.abspath(os.path.split(__file__)[0])
        self.output_dir = config_root
        self.exp_name = 'outputs_'+nowTime
        
        data_root = "/root/visDrone2019/"
        # 在这里找的 
        self.data_dir = data_root
        self.train_ann = '/root/visDrone2019/train.json'
        self.val_ann = '/root/visDrone2019/val.json'

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
