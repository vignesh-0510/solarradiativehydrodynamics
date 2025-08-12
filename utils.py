import wandb
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class WandbLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_dict = {k: v for k, v in logs.items()}
        wandb.log(metrics_dict, step=epoch)