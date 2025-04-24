import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('weights/OminiCrater_DETR.pt')
    model.val(data='dataset/data.yaml',
              split='val', 
              imgsz=768,
              batch=8,
              save_json=True, 
              project='runs/val',
              name='exp',
              )
    
