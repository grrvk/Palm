import albumentations as A
import numpy as np

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255


def train_transforms(img_size):
    train_image_transform = A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.Rotate(limit=25),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ], is_check_shapes=False)

    return train_image_transform


def valid_transforms(img_size):

    valid_image_transform = A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ], is_check_shapes=False)

    return valid_image_transform
