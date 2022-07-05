import numpy as np


def check_image_file(filename: str):
    return any(
        filename.endswith(extension)
        for extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
        ]
    )

def uint2single(img):
    return np.float32(img / 255.0)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())
