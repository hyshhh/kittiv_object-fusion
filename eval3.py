import mayavi.mlab as mlab
from test import visualization
from kitti_object import kitti_object,nms
from viz_util import draw_lidar
import cv2
from PIL import Image
import time
def get_fusion(objects,detects, calib, depth=None):              ########################################################################
    boxes=[]
    scores=[]
    keep=[]
    fusion=[]
    for obj in objects: 
      if obj.score>0.5	:
        if obj.type == "Car":
            boxes.append([int(obj.xmin),int(obj.ymin),int(obj.xmax),int(obj.ymax),obj.score])

    return boxes
import numpy as np

def compute_iou(box1, box2):
    """ 计算两个框的IoU """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(gt_boxes, pred_boxes, iou_thresh=0.5):
    """ 计算单类别 AP（预测框必须包含 score） """
    if pred_boxes == None:
        return 0.0
    
    # 按 score 降序排列预测框
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    detected = [False] * len(gt_boxes)

    for i, pred in enumerate(pred_boxes):
        pred_box = pred[:4]
        ious = [compute_iou(pred_box, gt) for gt in gt_boxes]
        max_iou = max(ious) if ious else 0
        max_idx = np.argmax(ious) if ious else -1

        if max_iou >= iou_thresh and 0 <= max_idx < len(detected) and not detected[max_idx]:

            tp[i] = 1
            detected[max_idx] = True
        else:
            fp[i] = 1

    # 累加 TP 和 FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precision = cum_tp / (cum_tp + cum_fp )
    recall = cum_tp / (len(gt_boxes) )
    print(precision)
    print(recall)
    # 插值 AP：∑(recall[i] - recall[i-1]) * precision[i]
    ap = 0.0
    for i in range(0, len(precision)):
       if i==0:                                             #计算最初一个recall的面积
          ap +=recall[i ]*precision[i]
       else:
          ap += (recall[i] - recall[i - 1]) * precision[i]

       
    ap = round(ap, 4)
    return ap
def get_label(labels_):
        labels=[]
        for label in labels_:
           if label.type=="Car":
             labels.append([int(label.xmin),int(label.ymin),int(label.xmax),int(label.ymax)])
        return labels
class eval1:
    # data_idx: determine data_idx
    def __init__(self, data_idx, root_dir=r'/home/hys/OpenPCDet/kitti_object_vis-master/data/object'):
        
        data=kitti_object(root_dir=root_dir)
        calib = data.get_calibration(data_idx)
        label = data.get_label_objects(data_idx)
        self.D2=data.get_detects(data_idx)
        self.calib=calib 
        self.D3=data.get_detects3D(data_idx)
        self.fusion=get_fusion(self.D2,self.D3,self.calib)
        self.labels=label
        self.labels_=get_label(self.labels)
    def evalution(self):
       gt_boxes=self.labels_
       pred_boxes=self.fusion
       return gt_boxes,pred_boxes
        
        

if __name__ == '__main__':
    x = 500      # 设置验证集数量
    flag=0
    MAP = 0.0
    gt_boxes=[]
    pred_boxes=[]
    print("#########################")
    print(" 开始评估 2D+3D 融合目标检测系统 ")
    print(" 验证集样本数量：", x)
    print("#########################\n")
    for i in range(x):
        eval1_instance = eval1(data_idx=i)  # 遍历每个样本
        a,b= eval1_instance.evalution()     # 调用评估方法
        if a is not None:
          gt_boxes+=a
        if b is not None:
          pred_boxes+=b
    AP=compute_ap(gt_boxes, pred_boxes, iou_thresh=0.5)    
        

    print("#########################")
    print(f"Car 类别的平均 AP（mAP）为: {AP:.4f}")
    print("评估完成。")

  

