from tqdm import tqdm
import torch
import numpy as np


def train_instance(
        model,
        train_dataloader,
        device,
        optimizer,
        processor,
        metric,
        num_classes=1
):
    print('Training instance')
    model.train()
    train_running_loss = []

    prog_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    for i, batch in enumerate(prog_bar):
        optimizer.zero_grad()

        outputs = model(
            pixel_values=batch['pixel_values'].to(device),
            mask_labels=[mask_label.to(device) for mask_label in batch['mask_labels']],
            class_labels=[class_label.to(device) for class_label in batch['class_labels']],
            pixel_mask=batch['pixel_mask'].to(device)
        )

        loss = outputs.loss
        train_running_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        target_sizes = [(image.shape[1], image.shape[2]) for image in batch['orig_images']]
        pred_maps = processor.post_process_instance_segmentation(
            outputs, target_sizes=target_sizes
        )
        binary_prediction_maps = []

        for p_map in pred_maps:
            binary_prediction_maps.append(np.array(p_map['segmentation'].cpu().detach().numpy(), dtype='uint8'))

        metric.add_batch(references=batch['orig_masks'], predictions=binary_prediction_maps)

    train_loss = sum(train_running_loss)/len(train_running_loss)
    metric = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)
    return train_loss, metric


def validate_instance(
        model,
        valid_dataloader,
        device,
        processor,
        metric,
        num_classes=1
):
    print('Validation instance')
    model.eval()
    valid_running_loss = []

    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader,
            total=(len(valid_dataloader)),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for i, batch in enumerate(prog_bar):

            outputs = model(
                pixel_values=batch['pixel_values'].to(device),
                mask_labels=[mask_label.to(device) for mask_label in batch['mask_labels']],
                class_labels=[class_label.to(device) for class_label in batch['class_labels']],
            )

            loss = outputs.loss
            valid_running_loss.append(loss.item())

            target_sizes = [(image.shape[1], image.shape[2]) for image in batch['orig_images']]
            pred_maps = processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )
            binary_prediction_maps = []

            for p_map in pred_maps:
                binary_prediction_maps.append(np.array(p_map['segmentation'].cpu().detach().numpy(), dtype='uint8'))

            metric.add_batch(references=batch['orig_masks'], predictions=binary_prediction_maps)

    valid_loss = sum(valid_running_loss)/len(valid_running_loss)
    metric = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)
    return valid_loss, metric



