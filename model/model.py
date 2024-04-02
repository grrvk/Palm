from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor)
from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
    AutoImageProcessor
)

def load_model(num_classes=2):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, do_normalize=False, do_rescale=False, do_resize=False
        #reduce_labels=True,
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        'facebook/mask2former-swin-tiny-ade-semantic',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    '''model = Mask2FormerForUniversalSegmentation.from_pretrained(
        'facebook/mask2former-swin-tiny-coco-instance',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )'''

    return model, preprocessor
