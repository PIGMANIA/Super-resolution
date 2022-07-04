import os
import hydra
import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
import numpy as np

from tqdm import tqdm
from utils import check_image_file


class Tester:
    def __init__(self, cfg, gpu):
        cudnn.benchmark = True

        self.gpu = gpu
        self.image_path = cfg.test.common.image_path
        self.save_path = cfg.test.common.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self._init_model(cfg.models)
        self._load_state_dict(cfg.models)
        self.test()

    def _init_model(self, model):
        if model.name.lower() == "scunet":
            from archs.SCUNet.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

        elif model.name.lower() == "realesrgan":
            from archs.RealESRGAN.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

        elif model.name.lower() == "bsrgan":
            from archs.BSRGAN.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

        elif model.name.lower() == "edsr":
            from archs.EDSR.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

    def _load_state_dict(self, model):
        if model.generator.path:
            ckpt = torch.load(
                model.generator.path, map_location=lambda storage, loc: storage
            )
            if len(ckpt) == 3:
                self.generator.load_state_dict(ckpt["g"])
            else:
                self.generator.load_state_dict(ckpt)

    def test(self):
        def preprocess(img):
            # uInt8 -> float32로 변환
            x = np.array(img).astype(np.float32)
            x = x.transpose([2, 0, 1])
            # Normalize x 값
            x /= 255.0
            # 넘파이 x를 텐서로 변환
            x = torch.from_numpy(x)
            # x의 차원의 수 증가
            x = x.unsqueeze(0)
            # x 값 반환
            return x

        def postprocess(tensor):
            x = tensor.mul(255.0).cpu().numpy().squeeze(0)
            x = np.array(x).transpose([1, 2, 0])
            x = np.clip(x, 0.0, 255.0).astype(np.uint8)
            x = pil_image.fromarray(x)
            return x

        self.generator.eval()
        images = []

        if os.path.isdir(self.image_path):
            for img in os.listdir(self.image_path):
                if check_image_file(img):
                    images.append(os.path.join(self.image_path, img))
        elif os.path.isfile(self.image_path):
            images.append(self.image_path)
        else:
            raise ValueError("Neither a file or directory")

        for path in tqdm(images):
            img = pil_image.open(path).convert("RGB")
            lr = preprocess(img).to(self.gpu)

            with torch.no_grad():
                preds = self.generator(lr)

            postprocess(preds).save(os.path.join(self.save_path, path.split("/")[-1]))


@hydra.main(config_path="./archs/SCUNet/configs", config_name="test.yaml")
def main(cfg):
    Tester(cfg, 0)


if __name__ == "__main__":
    main()
