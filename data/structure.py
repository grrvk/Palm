import torch
from torch.utils.data import Dataset
import numpy as np


def collate_fn(batch):
    inputs = list(zip(*batch))
    pixel_values = torch.stack([example["pixel_values"] for example in inputs[0]])
    pixel_mask = torch.stack([example["pixel_mask"] for example in inputs[0]])
    mask_labels = [example["mask_labels"] for example in inputs[0]]
    class_labels = [example["class_labels"] for example in inputs[0]]
    return {
        "image": inputs[1],
        "mask": inputs[2],
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

def collate_fn2(batch, image_processor):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    batch = image_processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors='pt',
    )
    batch['orig_image'] = inputs[2]
    batch['orig_mask'] = inputs[3]
    return batch


class ImageSegmentationDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB")).astype('uint8')
        class_id_map = np.array(self.dataset[idx]["annotation"])[..., 0].astype('float32')
        orig_image=image.copy()
        instance_seg = np.array(self.dataset[idx]["annotation"])[..., 1]
        orig_mask = instance_seg.copy()
        class_labels = np.unique(class_id_map)
        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            (image, instance_seg) = (transformed["image"], transformed["mask"])

            image = image.transpose(2, 0, 1)

        '''if class_labels.shape[0] == 1 and class_labels[0] == 0:
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:'''

        inputs = self.processor(
                [image],
                [instance_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt",
        )
        inputs = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()
        }
        return inputs, orig_image, orig_mask
