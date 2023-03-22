import argparse
from train import train_model
from detect import detect_faces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'detect'], help='Train or detect mode')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to the pretrained weights')
    parser.add_argument('--output_path', type=str,
    default='output', help='Path to save the output results')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--img_size', type=int, default=416, help='Size of input images')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='Object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for non-maximum suppression')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'detect':
        detect_faces(args)

if __name__ == '__main__':
    main()
