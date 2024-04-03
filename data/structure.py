import torch
from torch.utils.data import Dataset
import numpy as np


def collate_fn(examples):
    orig_images = [example["orig_image"] for example in examples]
    orig_masks = [example["orig_mask"] for example in examples]
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]
    return {
        "orig_images": orig_images,
        "orig_masks": orig_masks,
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }


class ImageSegmentationDataset(Dataset):
    def __init__(self, dataset, processor, type, transform=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB"))

        instance_seg = np.array(self.dataset[idx]["annotation"])[..., 1]
        class_id_map = np.array(self.dataset[idx]["annotation"])[..., 0]
        class_labels = np.unique(class_id_map)
        mask = instance_seg if self.type == 'instance' else class_id_map
        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            (image, mask) = (transformed["image"], transformed["mask"])

            image = image.transpose(2, 0, 1)
        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
            inputs['orig_image'] = image
            inputs['orig_mask'] = mask
        else:
            if self.type == 'instance':
                inputs = self.processor(
                    [image],
                    [mask],
                    instance_id_to_semantic_id=inst2class,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    [image],
                    [mask],
                    return_tensors="pt"
                )
            inputs = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()
            }
            inputs['orig_image'] = image
            inputs['orig_mask'] = mask
        return inputs
