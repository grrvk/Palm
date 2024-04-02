import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor,
                          MaskFormerForInstanceSegmentation,
                          MaskFormerImageProcessor)


def load_model(device, folder_name='out', name='model_out'):
    path = os.path.join(folder_name, name)
    '''model = Mask2FormerForUniversalSegmentation.from_pretrained(
        os.path.join(path, 'final_model')
    ).to(device)

    processor = Mask2FormerImageProcessor.from_pretrained(
        os.path.join(path, 'final_processor')
    )'''

    model = MaskFormerForInstanceSegmentation.from_pretrained(
        os.path.join(path, 'final_model')
    ).to(device)

    processor = MaskFormerImageProcessor.from_pretrained(
        os.path.join(path, 'final_processor')
    )
    return model, processor


def save_model(model, processor, folder_name='out', name='model_out'):
    out_dir = os.path.join(folder_name, name)
    out_final_model = os.path.join(out_dir, 'final_model')
    out_final_processor = os.path.join(out_dir, 'final_processor')

    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_final_model, exist_ok=True)
    os.makedirs(out_final_processor, exist_ok=True)

    processor.do_normalize = True
    processor.do_resize = True
    processor.do_rescale = True

    model.save_pretrained(out_final_model)
    processor.save_pretrained(out_final_processor)


def plot_loss_miou(epochs, train_loss, valid_loss, train_miou, valid_miou, folder_name='out'):
    os.makedirs(folder_name, exist_ok=True)
    out_dir = os.path.join(folder_name)

    x = np.arange(1, epochs+1)
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, valid_loss, label='validation loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train/validation loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    plt.plot(x, train_miou, label='train miou')
    plt.plot(x, valid_miou, label='validation miou')
    plt.xlabel("epoch")
    plt.ylabel("miou")
    plt.title("Train/validation miou")
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))

