import glob
import random

import numpy as np
from datasets import Dataset, Image
from matplotlib import pyplot as plt


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


def check_dataset_annotations(dataset):
    index = random.randint(0, len(dataset)-1)
    image = dataset[index]["image"]
    image = np.array(image.convert("RGB"))
    annotation = dataset[index]["annotation"]
    annotation = np.array(annotation)
    plt.figure(figsize=(15, 5))
    for plot_index in range(3):
        if plot_index == 0:
            plot_image = image
            title = "Original"
        else:
            plot_image = annotation[..., plot_index - 1]
            print(np.unique(plot_image))
            title = ["Class Map (R)", "Instance Map (G)"][plot_index - 1]

        plt.subplot(1, 3, plot_index + 1)
        plt.imshow(plot_image)
        plt.title(title)
        plt.axis("off")
    plt.show()
