import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils import save_model, plot_loss_miou, dump_metrics_log, delete_metrics
import evaluate
from model.engine_instance import train_instance, validate_instance
from model.engine_semantic import train_semantic, validate_semantic
from model.model import load_instance_model, load_semantic_model
from data.dataset import get_datasets, get_data_loaders
from datetime import datetime


def run_train(train_type, train_data, valid_data, epochs, batch_size, lr=5e-5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, processor = load_instance_model() if train_type == 'instance' else load_semantic_model()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_dataset, valid_dataset = get_datasets(train_data, valid_data,
                                                processor, train_type, img_size=[512, 512])

    train_dataloader, valid_dataloader = get_data_loaders(train_dataset, valid_dataset,
                                                          batch_size)

    scheduler = MultiStepLR(
        optimizer, milestones=[50], gamma=0.1)

    train_loss, train_miou = [], []
    valid_loss, valid_miou = [], []

    delete_metrics(train_type)
    metric = evaluate.load("mean_iou")

    for epoch in range(epochs):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_metric = (
            train_instance(model, train_dataloader, device, optimizer, processor, metric)
            if train_type == 'instance' else
            train_semantic(model, train_dataloader, device, optimizer, processor, metric))

        valid_epoch_loss, valid_epoch_metric = (
            validate_instance(model, valid_dataloader, device, processor, metric)
            if train_type == 'instance' else
            validate_semantic(model, valid_dataloader, device, processor, metric))

        total_metrics = {'train': train_epoch_metric, 'validation': valid_epoch_metric}

        train_loss.append(train_epoch_loss)
        train_miou.append(train_epoch_metric['mean_iou'])
        valid_loss.append(valid_epoch_loss)
        valid_miou.append(valid_epoch_metric['mean_iou'])

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch mIOU: {train_epoch_metric['mean_iou']:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},",
            f"Valid Epoch mIOU: {valid_epoch_metric['mean_iou']:4f}"
        )

        scheduler.step()
        dump_metrics_log(train_type, epoch + 1, datetime.now().time(), total_metrics)

    plot_loss_miou(train_type, epochs, train_loss, valid_loss, train_miou, valid_miou)
    save_model(model, processor, train_type)
    print('TRAINING COMPLETE')
