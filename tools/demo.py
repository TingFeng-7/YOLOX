#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform,ValTransform_32scaled
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, get_vis_boxes_score_cls, postprocess_soft_numpy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam,还有image-full"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold") 
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument('--class_agnostic', default='False', action='store_true', help='class-agnostic NMS')# add 
    parser.add_argument('--soft_nms', default='False', action='store_true', help='open soft-NMS')# add
    parser.add_argument('--json_folder','-j', type=str, help='设置保存预测结果的根目录')# add
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


# add score、自适应类别名
def save_result_sajson_auto(curr_image_bbox_class, curr_image_bboxes, scores, class_name, shape, name, save_folder):
    import json
    result_json={}
    #---
    result_json['imageName'] = name
    result_json['imgHeight'] = shape[0]
    result_json['imgWidth'] = shape[1]

    class_name_to_ids ={}
    for cls in class_name:
        result_json[cls] = []
    for k,v in enumerate(class_name):
        class_name_to_ids[v] = k
    # add
    for i in range(len(curr_image_bboxes)):
        cur_cls = curr_image_bbox_class[i]
        # cls_id = class_name_to_ids[cur_cls]
        if cur_cls == 'text':
            result_json[cur_cls].append({"location":curr_image_bboxes[i],'scores':scores[i], "content":''})
        else:
            result_json[cur_cls].append({"location":curr_image_bboxes[i], 'scores':scores[i]})
    os.makedirs(save_folder, exist_ok=True)#确定存在
    name = name[:-4] + '.json'
    json_name = os.path.join(save_folder, name)
    with open(json_name,'w', encoding='utf-8') as fw: 
        # json.dump(result_json, fw, indent=2)
        json.dump(result_json, fw)
        logger.info('{} saved'.format(json_name))

class Predictor(object):
    def __init__(
        self,
        model,
        args,#add
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.args = args
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        # self.agnostic = exp.agnostic
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform_32scaled(legacy=legacy) # change
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda() #测试尺寸
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size) # 前处理
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type()) # 解码输出 预测框scale 到 原图尺寸
            # print(f'parm:{self.args.class_agnostic}')
            if self.args.soft_nms == True:
                # logger.info('soft_nms')
                outputs = postprocess_soft_numpy(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
            #AGNOSTIC 默认 hardnms + class_aware
            elif self.args.class_agnostic == True:
                # logger.info('enter class_agnostic')
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
            #AWARE 默认 hardnms + class_aware
            else:
                # logger.info('enter class_aware')
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=False
                )
            logger.info("infer time : {:.4f}s".format(time.time() - t0))

        return outputs, img_info, self.args.soft_nms


    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        #hard nms [bbox_4, obj_1, score_1, clsid_1 ] [nx7]
        #soft nms [bbox_4, cls_id_1, score_1 ] [nx6]
        # 两个output略有不同

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        if output.shape[1] == 7:
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
        elif output.shape[1] == 6:
            cls = output[:, -2]
            scores = output[:, -1]
        # tingfeng
        # vis_res ,box_list, label_list, scores = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        vis_res ,box_list, label_list, scores = get_vis_boxes_score_cls(img, \
            bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, box_list, label_list, scores


# tinfeng add 一次性对 27 or 36 个 应用 进行测试
def image_full(predictor, vis_folder, path, current_time, save_result, json_folder):
        logger.info('文件目录名{}'.format(json_folder))
        logger.info('父名{}'.format(vis_folder))
        app_folders = [x for x in os.listdir(path) if x[0:4] <= '0027'] #控制数量 wtf
        app_folders = [x for x in os.listdir(path) if x[0:5] == 'label'] #控制数量 wtf
        app_folders.sort()
        logger.info(app_folders)

        for app_folder in app_folders:
            save_sa_folder = os.path.join(vis_folder, json_folder,"Element_grabbing","grabbing_evaluation", app_folder,'grabbing_predict') #2.保存结果的目录 要与测试的对齐的话
            save_vis_folder = os.path.join(vis_folder, json_folder,"Element_grabbing","grabbing_evaluation", app_folder,'garbbing_debug')
            image_demo(predictor, save_vis_folder, os.path.join(path, app_folder), current_time, save_result, save_sa_folder)
            # image_demo(predictor, save_vis_folder, os.path.join(path, app_folder, 'imgs'), current_time, save_result, save_sa_folder)
            

def image_demo(predictor, vis_folder, path, current_time, save_result, json_folder):
    # input: directory or 
    
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]

    files.sort()
    from tqdm import tqdm
    for image_name in tqdm(files):
        outputs, img_info, soft_nms = predictor.inference(image_name)
        #output[0] 因为batch为1 只推一张图
        logger.info(f'后处理后 tensor 尺寸 {outputs[0].shape}')
        #visualize Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        result_image, box_list, labels_list, scores  = predictor.visual(outputs[0], img_info, predictor.confthre)

        #wtf add 是否保存sa
        if json_folder != None:
            # save_result_sajson(labels_list, box_list, [img_info['height'], img_info['width']], img_info['file_name'],json_folder) 
            save_result_sajson_auto(labels_list, box_list, scores,predictor.cls_names, [img_info['height'], img_info['width']], img_info['file_name'],json_folder) 
            #2. 画debug ,建立图片保存位置
            # os.makedirs(vis_folder, exist_ok=True)             
            # save_file_name = os.path.join(vis_folder, os.path.basename(image_name))
            # logger.info("Saving detection result in {}".format(save_file_name)) #保存为文件 2.画debug
            # cv2.imwrite(save_file_name, result_image)#cv : mat
        #wtf end

        # 是否生成带有预测框的图片
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )# 以当前时间命名
            os.makedirs(save_folder, exist_ok=True)             
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name)) #保存为文件
            cv2.imwrite(save_file_name, result_image)#cv : mat

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    try:
        logger.info("EXP CLASS: {}".format(exp.cls_names))
    except:
        exp.cls_names=''
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    # vis_folder = None
    vis_folder = os.path.join(file_name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)
    # if args.save_result:
    #     vis_folder = os.path.join(file_name, "vis_res")
    #     os.makedirs(vis_folder, exist_ok=True)
    #不管是否保存结果 都需要vis_folder

    if args.trt:
        args.device = "gpu"

    logger.info("COMMAND LINE Args: {}".format(args))
    logger.info("json_folder: {}".format(args.json_folder))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    #TINGFENG ADD, 初始化送进去 coco_classes，
    #实验文件 可配置 cls_names
    if exp.cls_names == '':#如果文件里没指定，默认还是coco
        predictor = Predictor(
        model, args, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
        )
    else:
        predictor = Predictor(#如果文件指定了，送入自定义
        model, args, exp, exp.cls_names, trt_file, decoder,
        args.device, args.fp16, args.legacy,
        )
    # END
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result, args.json_folder)
    elif args.demo == "image-full" :# tingfeng add
        image_full(predictor, vis_folder, args.path, current_time, args.save_result, args.json_folder)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
