from .base import Losses

def get_losses(configs):
    loss = Losses(configs)
    return loss.select_loss()