#!/bin/bash

python tools/demo.py image-full \
 -f /home/tingfengwu/test09_0010_v1.5_filtered_linebox/config.py  \
 -c /home/tingfengwu/test09_0010_v1.5_filtered_linebox/outputs_2022-12-21-08-41-11/best_ckpt.pth \
 --path /data/screen_analysis/0002_sa_task/0010_v1.5_filtered_linebox_yolox/ \
 --json_folder 20230106_softnms_yolox-nano-fpn4  \
 --nms 0.5 --conf 0.1 --soft_nms

