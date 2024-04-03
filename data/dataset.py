from torch.utils.data import DataLoader
from data.transforms import train_transforms, valid_transforms
from data.structure import ImageSegmentationDataset, collate_fn

def get_datasets(train_data, valid_data, preprocessor, type, img_size):

    train_transform = train_transforms(img_size)
    validation_transform = valid_transforms(img_size)

    train_dataset = ImageSegmentationDataset(
        dataset=train_data,
        processor=preprocessor,
        transform=train_transform,
        type=type
    )

    valid_dataset = ImageSegmentationDataset(
        dataset=valid_data,
        processor=preprocessor,
        transform=validation_transform,
        type=type
    )

    return train_dataset, valid_dataset


def get_data_loaders(train_dataset, valid_dataset, batch_size):
    #collate_func = partial(collate_fn, image_processor=processor)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_data_loader, valid_data_loader
