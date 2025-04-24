import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import time
from time import sleep   

   
if __name__ == '__main__':    
    model = RTDETR('ultralytics/cfg/models/OminiCrater_DETR.yaml')
    # model.load('/home/lz/work_dir/RTDETR-20241118/RTDETR-main/weights/rtdetr-r18.pt') # loading pretrain weights  
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=768,
                epochs=100,
                batch=4,
                workers=0,
                device='0', 
                project='runs/train',
                name='exp',
                ) 
