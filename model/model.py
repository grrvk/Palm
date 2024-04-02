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
    preprocessor = MaskFormerImageProcessor(
        #reduce_labels=True,
        size=(512, 512),
        ignore_index=255,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    model_name = "facebook/maskformer-swin-base-ade"
    config = MaskFormerConfig.from_pretrained(model_name)
    id2label = {0: 'bg', 1: 'Table'}
    label2id = {'bg': 0, 'Table': 1}
    config.id2label = id2label
    config.label2id = label2id
    model = MaskFormerForInstanceSegmentation(config)
    base_model = MaskFormerModel.from_pretrained(model_name)
    model.model = base_model

    return model, preprocessor
