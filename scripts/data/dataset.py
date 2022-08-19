import os
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset

from data.data_prepare import DataPrepare
from utils import check_image_file


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
