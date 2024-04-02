import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils import save_model, plot_loss_miou
import evaluate
from model.engine import train, validate
from model.model import load_model
from data.dataset import get_datasets, get_data_loaders


def run(train_data, valid_data, epochs, batch_size, lr=5e-5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, processor = load_model()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_dataset, valid_dataset = get_datasets(train_data, valid_data,
                                                processor, img_size=[512, 512])

    train_dataloader, valid_dataloader = get_data_loaders(train_dataset, valid_dataset,
                                                          batch_size)

    scheduler = MultiStepLR(
        optimizer, milestones=[50], gamma=0.1)

    train_loss, train_miou = [], []
    valid_loss, valid_miou = [], []

    metric = evaluate.load("mean_iou")

    for epoch in range(epochs):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            2,
            processor,
            metric
        )

        valid_epoch_loss, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            1,
            processor,
            metric
        )

        train_loss.append(train_epoch_loss)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_miou.append(valid_epoch_miou)

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )

        scheduler.step()

    plot_loss_miou(epochs, train_loss, valid_loss, train_miou, valid_miou)
    save_model(model, processor)
    print('TRAINING COMPLETE')