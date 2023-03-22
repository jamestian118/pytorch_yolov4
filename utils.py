import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms

def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5):
    # NMS 实现代码，这部分需要根据完整的 YOLOv4 实现进行补充
    pass

def plot_detections(img, detections, output_path):
    for detection in detections:
        if detection is not None:
            for det in detection:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = det
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_path, 'detections.jpg'), img)

def plot_losses(losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_path, 'losses.png'))
