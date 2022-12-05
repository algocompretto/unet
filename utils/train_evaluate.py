import torch
from torch import nn
import utils.distributed_utils as utils
from utils.dice_loss import dice_loss, build_target


def criterion(
    x,
    target,
    loss_weight=None,
    num_classes: int = 2,
    dice: bool = True,
    ignore_index: int = -100,
):
    print(x.shape)
    print(target.shape)
    loss = nn.functional.cross_entropy(
        x, target, weight=loss_weight, ignore_index=ignore_index, 
    )
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss


def evaluate(model, data_loader, device, num_classes: int):
    model.eval()
    conf_matrix = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = "Evaluation: "
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            conf_matrix.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
        conf_matrix.reduce_from_all_processes()
        dice.reduce_from_all_processes()
    return conf_matrix, dice.value.item()


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    num_classes,
    epochs,
    lr_scheduler,
    print_freq=5,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}/{}]".format(epoch, epochs)

    if num_classes == 2:
        # Set the loss weight of background and foreground in cross_entropy
        loss_weight = torch.as_tensor([1.0, 1.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(
                output, target, loss_weight, num_classes=num_classes, ignore_index=255
            )

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
):
    assert num_step > 0 and epochs > 0

    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Returns a learning rate multiplication factor based on the number of steps,
            Note that before the training starts, pytorch will call the lr_scheduler.step() method in advance
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # During the warmup process, the lr multiplication factor changes from warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (
                1
                - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)
            ) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
