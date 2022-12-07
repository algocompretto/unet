import torch
from torch import Tensor


def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    """Dice coefficient function. For theoretical background, see https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        input (Tensor): The predicted tensor.
        target (Tensor): The ground truth mask.
        reduce_batch_first (bool, optional): Whether to reduce the batch first or not. Defaults to False.
        epsilon (_type_, optional): Epsilon constant. Defaults to 1e-6.

    Raises:
        ValueError: Raises ValueError if the input tensor has no batch dimension.

    Returns:
        _type_: Calculated loss tensor using the Dice metric.
    """
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
        )

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    """Multiclass Dice coefficient. Same thing as the `dice_coeff` function, but for multiple classes.

    Args:
        input (Tensor): The predicted tensor.
        target (Tensor): The ground truth mask.
        reduce_batch_first (bool, optional): Whether to reduce the batch first or not. Defaults to False.
        epsilon (_type_, optional): Epsilon constant. Defaults to 1e-6.

    Returns:
        _type_: Calculated loss tensor using the Dice metric.
    """
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon
        )

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """Calculates the Dice loss for the current prediction and ground truth.

    Args:
        input (Tensor): The predicted tensor.
        target (Tensor): The ground truth tensor
        multiclass (bool, optional): Boolean to check whether the Dice loss is for a multiclass problem or not. Defaults to False.

    Returns:
        _type_: Returns a value which corresponds to the Dice loss for the given input and target values.
    """
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
