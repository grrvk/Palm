import torch
from utils import load_model
import random
import numpy as np
import matplotlib.pyplot as plt


def visualize_instance_seg_mask(mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    labels = np.unique(mask)
    print(labels)

    label2color = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels
    }

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]

    image = image / 255
    return image


def dataset_inference(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device)

    for i, entry in enumerate(dataset):
        print(f"Image {i+1}")
        image = entry["image"].convert("RGB")
        inf(image, model, processor)


def inf(image, model=None, processor=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not model or not processor:
        model, processor = load_model(device)

    inputs = processor(images=image, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    '''result = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.4,
        target_sizes=[[image.size[1], image.size[0]]]
    )[0]'''

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[[image.size[1], image.size[0]]]
    )[0]

    print(np.unique(result.cpu().detach().numpy()))
    plt.imshow(result.cpu().detach().numpy())
    plt.axis("off")
    plt.show()

    plt.imshow(image)
    plt.axis("off")
    plt.show()


    '''instance_seg_mask = result["segmentation"].cpu().detach().numpy()
    print(f"Final mask shape: {instance_seg_mask.shape}")
    print("Segments Information...")
    for info in result["segments_info"]:
        print(f"  {info}")

    instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask)
    plt.figure(figsize=(10, 10))

    for plot_index in range(2):
        if plot_index == 0:
            plot_image = image
            title = "Original"
        else:
            plot_image = instance_seg_mask_disp
            title = "Segmentation"

        plt.subplot(1, 2, plot_index + 1)
        plt.imshow(plot_image)
        plt.title(title)
        plt.axis("off")
        plt.show()'''

