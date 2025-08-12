# create DNN that models log(rho) and log(T) stratification of BiFrost model
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from scales import scales
from readBifrost import readBifrost
from saha_tf import saha
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
import toml
import wandb
from utils import WandbLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models.strat import model as strat_model


# === adjustable parameters ===
parser = argparse.ArgumentParser(
    prog = 'strat',
    description = 'Train a neural network to Stratify z-axis of Bifrost model'
)
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'strat.keras',
                    help = 'neural network output model name')

args = parser.parse_args()

with open("config/strat.toml", "r") as f:
    config = toml.load(f, _dict=dict)

base_dir =       config['train_params']['base_dir'] # base directory for data
save_dir =       os.path.join(base_dir, 'outputs') # directory for saving data
image_dir =       os.path.join(base_dir, 'images') # directory for saving data
epochs =           config['train_params']['epochs']        # number of epochs
batch_size =       config['train_params']['batch_size']    # batch size
learning_rate =    config['train_params']['learning_rate'] # learning rate for optimizer

model_name    = args.output_model

# load BiFrost model
params = ['ux', 'uy', 'uz', 'lgr', 'lgp', 'lgtg']
apar, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, 1, True)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = apar.shape # number of grid points in each axis
ndim = apar.ndim - 1 # number of dimensions; par also has an axis for params
apar = np.moveaxis(apar,1,3)
input_scale, input_offset, output_scale, output_offset = scales()

# density stratification
lgr = np.mean(apar[:,:,:,:,3], axis=(0,1,2))

# temperature stratification
lgt = np.mean(apar[:,:,:,:,5], axis=(0,1,2))

def scaleOffset(input):
    """
    provide offset and scaling for input such that
    output = input * scale + offset
    is in the range -1 to +1
    """
    min = np.amin(input)
    max = np.amax(input)
    scale = 2.0 / (max - min)
    offset = -min * scale - 1.0
    return (input * scale + offset, scale, offset)

z_norm,   z_scale,   z_offset   = scaleOffset(zaxis)
lgr_norm, lgr_scale, lgr_offset  = scaleOffset(lgr)
lgt_norm, lgt_scale, lgt_offset = scaleOffset(lgt)
output_scale  = [1.0/lgr_scale, 1.0/lgt_scale]
output_offset = [-lgr_offset/lgr_scale, -lgt_offset/lgt_scale]


# neural network:
#   output is mean(log10(rho)), mean(log10(T))
#   all outputs are in physical coordinates
model = strat_model(z_scale, z_offset, output_scale, output_offset)

X = tf.convert_to_tensor(zaxis, dtype="float32")
Y = tf.convert_to_tensor(np.stack((lgr,lgt), axis = -1), dtype="float32")

lscale = [1e-1, 1e-2]

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.reduce_mean(tf.square(y_true - y_pred), axis = 0) / tf.square(lscale)
    )

wandb.login()

wandb_params = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "scheduler": "ReduceLROnPlateau",
    "Optimizer": "Adam",
}

wandb.init(
    name=config['wandb_params']['run_name'],
    config=wandb_params
)

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = loss_fn
)

check_point = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(save_dir, model_name),
    monitor = 'loss',
    save_best_only = True,
    # initial_value_threshold = 1e-3
)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=400, min_lr=1e-6, verbose=1)

model.fit(
    X,
    Y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 1,
    callbacks = [check_point, lr_scheduler, WandbLogger()],
)

# print neural network structure
model.summary()

# read best model and predict stratification
best_model = tf.keras.models.load_model(os.path.join(save_dir, model_name), compile=False)
artifact = wandb.Artifact('stratification-model', type='model')
artifact.add_file(os.path.join(save_dir, model_name))
wandb.log_artifact(artifact)

predicted = best_model.predict(zaxis)
bifrost = [lgr, lgt]
label   = ['lgr', 'lgt']

print('losses per atmospheric parameter')
wandb_metrics = {}
for i in range(len(label)):
    loss_val = np.mean((predicted[:,i] - bifrost[i])**2, axis = 0) / lscale[i]**2
    print(label[i], loss_val)
    wandb_metrics['loss_'+label[i]] = loss_val

wandb.log(wandb_metrics)

for i in range(len(label)):
    plt.plot(zaxis/1e3, bifrost[i], label = 'Bifrost '+label[i])
    plt.plot(zaxis/1e3, predicted[:,i], label = 'DNN '+label[i])
    plt.legend()
    plt.xlabel('z[km]')
    plt.savefig(os.path.join(base_dir,f'outputs/strat_{label[i]}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    artifact = wandb.Artifact(f'mae_{params[i]}.png', type='evaluation')
    artifact.add_file(os.path.join(image_dir, 'int2mod', f'mae_{params[i]}.png'))
    wandb.log_artifact(artifact)

wandb.finish()