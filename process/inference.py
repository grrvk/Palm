import torch
from utils import load_model
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


def visualize_instance_seg_mask(mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    labels = np.unique(mask)

    label2color = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels if label != -1.0
    }

    label2color.update({-1.0: (255, 255, 255)})

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]

    image = image / 255
    return image


def df_dataset_inference(dataset, inference_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device, inference_type)

    for i, entry in enumerate(dataset):
        print(f"Image {i + 1}")
        image = entry["image"].convert("RGB")
        if inference_type == 'instance':
            inference_instance(image, model, processor)
        else:
            inference_semantic(image, model, processor)


def dataset_inference(dataset_path: str, inference_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device, inference_type)
    data = [file for file in glob.glob(dataset_path)]
    for i, entry in enumerate(data):
        print(f"Image {i + 1}")
        image = Image.open(entry).convert("RGB")
        if inference_type == 'instance':
            inference_instance(image, model, processor)
        else:
            inference_semantic(image, model, processor)


def get_outputs(model_type, image, model=None, processor=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not model or not processor:
        model, processor = load_model(device, model_type)

    inputs = processor(images=image, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs, processor


def inference_instance(image, model=None, processor=None):
    outputs, processor = get_outputs('instance', image, model, processor)

    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.7,
        target_sizes=[[image.size[1], image.size[0]]]
    )[0]

    label_of_interest = 0
    instance_seg_mask = result["segmentation"].cpu().detach().numpy()
    print("Segments Information...")
    for info in result["segments_info"]:
        print(f"  {info}")
        if info.get('label_id') != label_of_interest:
            instance_seg_mask[instance_seg_mask == info.get('id')] = -1

    instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask)
    plt.figure(figsize=(10, 10))

    for plot_index in range(2):
        if plot_index == 0:
            plot_image = image
            title = "Original"
        else:
            plot_image = instance_seg_mask_disp
            title = "Instance segmentation"

        plt.subplot(1, 2, plot_index + 1)
        plt.imshow(plot_image)
        plt.title(title)
    plt.show()


def inference_semantic(image, model=None, processor=None):
    outputs, processor = get_outputs('semantic', image, model, processor)

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[[image.size[1], image.size[0]]]
    )[0]

    print(np.unique(result.cpu().detach().numpy()))
    plt.imshow(result.cpu().detach().numpy())
    plt.axis("off")
    plt.show()

    plt.imshow(image)
    plt.show()
