import os
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor,
                          MaskFormerForInstanceSegmentation,
                          MaskFormerImageProcessor)


def load_model(device, folder_name='model_out'):
    path = os.path.join(folder_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        os.path.join(path, 'final_model')
    ).to(device)

    processor = Mask2FormerImageProcessor.from_pretrained(
        os.path.join(path, 'final_processor')
    )

    '''model = MaskFormerForInstanceSegmentation.from_pretrained(
        os.path.join(path, 'final_model')
    ).to(device)

    processor = MaskFormerImageProcessor.from_pretrained(
        os.path.join(path, 'final_processor')
    )'''
    return model, processor


def save_model(model, processor, name='model_out'):
    out_dir = os.path.join(name)
    out_final_model = os.path.join(out_dir, 'final_model')
    out_final_processor = os.path.join(out_dir, 'final_processor')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_final_model, exist_ok=True)
    os.makedirs(out_final_processor, exist_ok=True)

    processor.do_normalize = True
    processor.do_resize = True
    processor.do_rescale = True

    model.save_pretrained(out_final_model)
    processor.save_pretrained(out_final_processor)
