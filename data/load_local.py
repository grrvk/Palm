import glob
from datasets import Dataset, Image


def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/mask_train/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/val/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/mask_val/*")
    valid_masks.sort()
    return train_images, train_masks, valid_images, valid_masks


def get_dataset(train_images, train_masks, valid_images, valid_masks):
    train_dataset = Dataset.from_dict({"image": train_images,
                                       "annotation": train_masks})
    train_dataset = train_dataset.cast_column("image", Image())
    train_dataset = train_dataset.cast_column("annotation", Image())

    val_dataset = Dataset.from_dict({"image": valid_images,
                                     "annotation": valid_masks})
    val_dataset = val_dataset.cast_column("image", Image())
    val_dataset = val_dataset.cast_column("annotation", Image())

    return train_dataset, val_dataset


