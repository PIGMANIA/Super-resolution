import os
import torchvision.transforms as transforms
import cv2
import numpy as np

from torch.utils.data import Dataset
from data.data_prepare import DataPrepare
from utils import check_image_file
from glob import glob

class Dataset(Dataset):
    def __init__(self, cfg):

        self.data_pipeline = DataPrepare(cfg)

        self.hrfiles = [
            os.path.join(cfg.train.dataset.train_dir, x)
            for x in os.listdir(cfg.train.dataset.train_dir)
            if check_image_file(x)
        ]

        self.len = len(self.hrfiles)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        hr = cv2.imread(self.hrfiles[index])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr, hr = self.data_pipeline.data_pipeline(hr)
        return self.to_tensor(lr), self.to_tensor(hr)

    def __len__(self):
        return self.len

class ValidDataset():
    def __init__(self, cfg):
        self.hr_dir = cfg.valid.dataset.hr_dir
        self.lr_dir = cfg.valid.dataset.lr_dir

        self.hrfiles = [
            os.path.join(self.hr_dir, x)
            for x in os.listdir(self.hr_dir)
            if check_image_file(x)
        ]
        self.lrfiles = [
            os.path.join(self.lr_dir, x)
            for x in os.listdir(self.lr_dir)
            if check_image_file(x)
        ]

        # assert len(self.hrfiles) == len(self.lrfiles), "hrfile != lrfile"

        self.scale = cfg.models.generator.scale
        self.len = len(self.hrfiles)

    def __getitem__(self, index):
        def bgr2rgb(src):
            return cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        hr = cv2.imread(self.hrfiles[index])
        hr_h, hr_w = hr.shape[:2]
        hr = bgr2rgb(hr)

        hr_filename = os.path.basename(self.hrfiles[index].split("/")[-1])
        hr_filename = os.path.splitext(hr_filename)[0]
        
        lr_files = glob(os.path.join(self.lr_dir, hr_filename + "*"))
        lrs = []
        lrnames = []
        for lr_file in lr_files:
            lrnames.append(lr_file.split("/")[-1])
            lr = cv2.imread(lr_file)
            lr = bgr2rgb(lr)
            lr_h, lr_w = hr.shape[:2]
        
            if hr_h == lr_h and hr_w == lr_w:
                lr = cv2.resize(lr, (hr_w//self.scale, hr_h//self.scale), cv2.INTER_CUBIC)
            lrs.append(lr)
    
        return lrs, hr, lrnames

    def __len__(self):
        return self.len