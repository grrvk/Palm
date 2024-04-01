from tqdm import tqdm
import torch
import numpy as np


def train(
        model,
        train_dataloader,
        device,
        optimizer,
        num_classes,
        processor,
        metric
):
    print('Training')
    model.train()
    train_running_loss = []

    prog_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    for i, batch in enumerate(prog_bar):
        optimizer.zero_grad()

        #print(batch['mask_labels'])
        #print(batch['class_labels'])
        outputs = model(
            pixel_values=batch['pixel_values'].to(device),
            mask_labels=[mask_label.to(device) for mask_label in batch['mask_labels']],
            class_labels=[class_label.to(device) for class_label in batch['class_labels']],
            pixel_mask=batch['pixel_mask'].to(device)
        )

        ##### BATCH-WISE LOSS #####
        loss = outputs.loss
        train_running_loss.append(loss.item())
        ###########################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################
        target_sizes = [(image.shape[0], image.shape[1]) for image in batch['image']]
        pred_maps = processor.post_process_instance_segmentation(
            outputs, target_sizes=target_sizes
        )
        pred_maps_2 = []
        pred_maps_un=[]
        batch_mask_un=[]

        for p_map in pred_maps:
            #pred_maps_un.append(np.unique(p_map))
            pred_maps_un.append(np.unique(p_map['segmentation'].cpu().detach().numpy()))
            pred_maps_2.append(np.array(p_map['segmentation'].cpu().detach().numpy(), dtype='uint8'))
            #print(np.array(p_map['segmentation'].cpu().detach().numpy()).shape)'''
        for b_mask in batch['mask']:
            batch_mask_un.append(np.unique(b_mask))
            #print(b_mask.shape)
        print(pred_maps_un)
        print(batch_mask_un)
        metric.add_batch(references=batch['mask'], predictions=pred_maps_2)

    ##### PER EPOCH LOSS #####
    train_loss = sum(train_running_loss)/len(train_running_loss)
    ##########################
    iou = metric.compute(num_labels=99, ignore_index=255, reduce_labels=True)['mean_iou']
    return train_loss, iou


def validate(
        model,
        valid_dataloader,
        device,
        num_classes,
        processor,
        metric
):
    print('Validating')
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

            '''target_sizes = [(image.shape[1], image.shape[2]) for image in batch['image']]
            pred_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )

            ##### BATCH-WISE LOSS #####
            loss = outputs.loss
            valid_running_loss.append(loss.item())
            ###########################
            metric.add_batch(references=batch['mask'], predictions=pred_maps)'''

    ##### PER EPOCH LOSS #####
    valid_loss = sum(valid_running_loss)/len(valid_running_loss)
    ##########################
    #iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)['mean_iou']
    return valid_loss, 0