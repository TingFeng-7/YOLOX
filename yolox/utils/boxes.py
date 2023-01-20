#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np
from loguru import logger
import torch
import torchvision
import copy
from torchvision.ops import box_iou
__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "postprocess_soft_numpy",
]

def np_1row_to_1cols(arr):
    return arr.reshape((len(arr), 1))

#tingfeng
def soft_nms_a(boxes, scores,nms_thr,score_thr, sigma =0.5,method = 'linear'):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # full_weight = np.ones(len(scores))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = areas.reshape(-1,1)
    scores = scores.reshape(-1,1)
    dets = np.concatenate((boxes, scores, areas) ,axis=1)#[4,1,1,1]
    #对得分进行排序 降序排列
    order = dets[np.argsort(dets[:,5])[::-1]]
    # order = scores.argsort()[::-1]
    #记录结果值，每次保存得分最高的那个框的索引，最后再用boxes[keep]取出相应框
    keep = []
    keep1 = np.empty((0,dets.shape[1]), int)
    #一直筛选到没有可用的框
    logger.info(f'before 框 {len(order)}')
    while order.size > 0:
        # print(order.size)
        #取得分最高的框的索引，因为order是升序，所以最后一位是得分最高的
        i = order[0]
        #保存得分最高的那个框的索引
        keep.append(i)

        order = order[1:,...]
        # IoU calculate ，det并没减少
        xx1 = np.maximum(i[0], order[:, 0])
        yy1 = np.maximum(i[1], order[:, 1])
        xx2 = np.minimum(i[2], order[:, 2])
        yy2 = np.minimum(i[3], order[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #永远排除当前
        ovr = inter / (i[-1] + order[:,-1] - inter)

        if method == 'gaus':
            weight = np.exp(-(ovr * ovr) / sigma)
        elif method == 'linear':
            weight = np.ones(ovr.shape)
            weight = weight - ovr
        # np.where返回元组
        index_after = np.where(ovr < nms_thr)[0]
        #筛选大于阈值的框，然后对其进行分数衰减
        soft_index = np.where(ovr >= nms_thr)[0] #返回的是ovr这里的位置

        if len(soft_index) == 0:
            #没有与其重复的框
            continue
        #有的话
        else:
            order[soft_index,-2] = order[soft_index,-2] * weight[soft_index]
            # logger.info(f'应被过滤的尺寸: {soft_index.size}')
            # logger.info(f'应被过滤的id: {order[soft_index+1]}')
            # logger.info(f'应被留下的id: {order[index_after + 1]}')

            # weight = np.exp(-(ovr[soft_index] * ovr[soft_index]) / 0.5)
            # 分开算在拼起来
            # 一起算
            part1 = order[index_after,:]
            save_id = np.where(order[soft_index,-2] >= score_thr)[0]
            part2 = order[soft_index,:][save_id]
            # 拼起来再重排序
            order = np.concatenate((part1 , part2), axis=0)
            order = order[np.argsort(order[:,-2])[::-1]]

    logger.info(f'after 长度{len(keep)}')
    keep = np.array(keep)
    # logger.info(np.sort(keep))
    # [4 bbox 1 cls-id 1 conf ]
    return keep[:,:-1]


# tingfeng numpy 实现
def postprocess_soft_numpy(prediction, num_classes, conf_thre=0.1, nms_thre=0.45, class_agnostic=True):
    #input : 网络输出 + 类别数量
    # first_conf = 0.05
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction): # [batch 是1就不会出错了]
        # If none are remaining => process next image
        # logger.info(f"batch :{i+1}")
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence 获取各类别的概率
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True) #每个类别都有概率 dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        # tingfeng
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred) 前景置信度、类别置信度、类别预测id。
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1) #拼位置 ，和置信度，和最好的id
        detections = detections[conf_mask]
        # shape (nums , 7)

    # logger.info((conf_mask==False).sum())
    # predict shape: torch.Size([1, 173400, 8])
    #1.numpy 版本 tesnor ---> numpy来计算
    n_prediction = detections
    x1 = n_prediction[:,0] - n_prediction[:,2]/2
    y1 = n_prediction[:,1] - n_prediction[:,3]/2
    x2 = n_prediction[:,0] + n_prediction[:,2]/2
    y2 = n_prediction[:,1] + n_prediction[:,3]/2

    x1 = np_1row_to_1cols(x1)
    x2 = np_1row_to_1cols(x2)
    y1 = np_1row_to_1cols(y1)
    y2 = np_1row_to_1cols(y2)
    front_obj = np_1row_to_1cols(n_prediction[:,4])
    class_conf = np_1row_to_1cols(n_prediction[:,5])
    class_pred = np_1row_to_1cols(n_prediction[:,6])
    # logger.info(f'Before NMS fillter shape: {class_conf.shape}')
    dets = np.concatenate([x1,y1,x2,y2,class_pred,front_obj*class_conf], axis=1)

    if class_agnostic == True:
        print('unknow')
        output = soft_nms_a(boxes = dets[:,:5] ,scores= dets[:,5],nms_thr=nms_thre, score_thr=conf_thre)
    else:
        print("aware")
        output = np.empty((0,dets.shape[1]))
        for cls_id in range(num_classes):
            tmp_dets = dets[np.where(dets[:,4]==cls_id)]
            if(len(tmp_dets)==0):
                continue
            one_output = soft_nms_a(boxes = tmp_dets[:,:5] ,scores= tmp_dets[:,5],nms_thr=nms_thre, score_thr=conf_thre)
            output = np.concatenate((output, one_output))
    output = torch.from_numpy(output)
    # logger.info(f'After NMS fillter shape: {output.shape}')
    output = torch.unsqueeze(output, dim=0)
    return output


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # logger.info(f'predict shape: {prediction.shape}')
    box_corner = prediction.new(prediction.shape)

    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True) #每个类别都有概率 dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        # logger.info('预测的前10行:{}'.format(image_pred[:10, 5: 5 + num_classes]))

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()#np.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用。
        # conf_mask 是一堆索引呢
        # logger.info(image_pred[:, 4] * class_conf.squeeze())
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred) 前景置信度、类别置信度、类别预测id。
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1) #拼位置 ，和置信度，和最好的id
        # logger.info("before fillter: {}".format(len(detections)))
        detections = detections[conf_mask]#取剩下的这些行

        # logger.info(f'before nms shape: {detections.shape}')
        if not detections.size(0):
            logger.info('当前检测框为0')
            continue

        if class_agnostic:
            # logger.info('class_agnostic')
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            # logger.info('class_aware')
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    # output: 一个多维数组
    # logger.info(f'output 类型: {type(output)}, output 形状: {output[0].shape}')
    return output

# ori: 计算[n x 4] 与 [m x 4] 之间的iou
def bboxes_iou_old(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1) # multiply
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1) # multiply
        #计算 left-top right-bottom ：areaA areaB
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

# new tingfeng
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        return box_iou(bboxes_a, bboxes_b) # 调的 torch 官方 api

    else:
        bboxes_a = cxcywh2xyxy(bboxes_a)
        bboxes_b = cxcywh2xyxy(bboxes_b)
        return box_iou(bboxes_a, bboxes_b)

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def cxcywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y