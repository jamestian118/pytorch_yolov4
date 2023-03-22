import torch
import cv2
import numpy as np
from model import YOLOv4
from torchvision.ops import nms
from utils import non_max_suppression, plot_detections

def detect_faces(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = YOLOv4().to(device)
    model.load_state_dict(torch.load(args.pretrained_weights))
    model.eval()

    # Read image
    img = cv2.imread(args.data_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))

    # Preprocess image
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Detect faces
    with torch.no_grad():
        detections = model(img_tensor)
        detections = non_max_suppression(detections, args.conf_thres, args.iou_thres)

    # Plot detections
    plot_detections(img, detections, args.output_path)
