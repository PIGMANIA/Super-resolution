import glob
import os
import torchvision.transforms as transforms
import cv2
import numpy as np

from torch.utils.data import Dataset
from data.data_prepare import DataPrepare
from utils import check_image_file, convert_rgb_to_y


class TrainDataset(Dataset):
    def __init__(self, train, model):
        self.data_pipeline = DataPrepare(train, model)
        self.hrfiles = [
            os.path.join(train.hr_dir, x)
            for x in os.listdir(train.hr_dir)
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


class TrainYUVDataset(Dataset):
    def __init__(self, train, model):
        self.data_pipeline = DataPrepare(train, model)
        assert model.generator.in_nc == 1, "input channel is not 1"

        self.hrfiles = [
            os.path.join(train.hr_dir, x)
            for x in os.listdir(train.hr_dir)
            if check_image_file(x)
        ]

        self.len = len(self.hrfiles)

    def __getitem__(self, index):
        hr = cv2.imread(self.hrfiles[index])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr, hr = self.data_pipeline.data_pipeline(hr)

        lr = convert_rgb_to_y(lr)
        hr = convert_rgb_to_y(hr)

        return np.expand_dims(lr / 255.0, 0), np.expand_dims(hr / 255.0, 0)

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self, test, model):
        super().__init__()
        self.scale = model.generator.scale
        self.lrfiles = [
            os.path.join(test.lr_dir, x)
            for x in os.listdir(test.lr_dir)
            if check_image_file(x)
        ]
        self.to_tensor = transforms.ToTensor()
        self.len = len(self.lrfiles)

    def __getitem__(self, index):
        lr = cv2.imread(self.lrfiles[index])
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        return self.to_tensor(lr)

    def __len__(self):
        return self.len


class ValidDataset(Dataset):
    def __init__(self, valid, model):
        super().__init__()
        self.scale = model.generator.scale
        self.hrfiles = [
            os.path.join(valid.hr_dir, x)
            for x in os.listdir(valid.hr_dir)
            if check_image_file(x)
        ]

        self.lrfiles = [
            os.path.join(valid.lr_dir, x)
            for x in os.listdir(valid.lr_dir)
            if check_image_file(x)
        ]

        self.hrfiles.sort()
        self.lrfiles.sort()

        self.len = len(self.hrfiles)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        hr = cv2.imread(self.hrfiles[index])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr = cv2.imread(self.lrfiles[index])
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        return self.to_tensor(lr), self.to_tensor(hr)

    def __len__(self):
        return self.len


class PiPalValidDataset:
    def __init__(self, valid, model):
        self.hr_dir = valid.hr_dir
        self.lr_dir = valid.lr_dir

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

        self.scale = model.generator.scale
        self.len = len(self.hrfiles)

    def __getitem__(self, index):
        def bgr2rgb(src):
            return cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        hr = cv2.imread(self.hrfiles[index])
        hr_h, hr_w = hr.shape[:2]
        hr = bgr2rgb(hr)

        hr_filename = os.path.basename(self.hrfiles[index].split("/")[-1])
        hr_filename = os.path.splitext(hr_filename)[0]
        lr_files = glob.glob(os.path.join(self.lr_dir, hr_filename + "*"))

        lrs = []
        lrnames = []
        for lr_file in lr_files:
            lrnames.append(lr_file.split("/")[-1])
            lr = cv2.imread(lr_file)
            lr = bgr2rgb(lr)
            lr_h, lr_w = hr.shape[:2]

            if hr_h == lr_h and hr_w == lr_w:
                lr = cv2.resize(
                    lr,
                    (hr_w // self.scale, hr_h // self.scale),
                    cv2.INTER_CUBIC,
                )
            lrs.append(lr)

        return lrs, hr, lrnames

    def __len__(self):
        return self.len
