import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('ultralytics/cfg/models/my/do/rtdetr-r50.yaml')
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[768, 768])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()