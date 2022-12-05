import torch
from torch import nn
import utils.distributed_utils as utils
from dice_loss import dice_loss, build_target


def criterion(
    x,
    target,
    loss_weight=None,
    num_classes: int = 2,
    dice: bool = True,
    ignore_index: int = -100,
):
    loss = nn.functional.cross_entropy(
        x, target, ignore_index=ignore_index, weight=loss_weight
    )
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss
