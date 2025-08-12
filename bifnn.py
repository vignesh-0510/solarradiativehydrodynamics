# train deep neural network directly on Bifrost simulations
# author: Christoph U.Keller, ckeller@nso.edu

import extract
from readBifrost import readBifrost
from scales import scales
import models.dnn as dnn
import plot
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import wandb
from utils import WandbLogger
import toml
import os

# parse arguments
parser = argparse.ArgumentParser(
    prog = 'bifnn',
    description = 'Train a deep neural network to reproduce a Bifrost simulation'
)
parser.add_argument('-y', '--deltaY',
                    type = int,
                    default = 0,
                    help = 'change in y extraction')
parser.add_argument('-s', '--step',
                    type = int,
                    default = 1,
                    help = 'step when comparing to data in x,y,z')

parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'model to start from')
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'bifnn.keras',
                    help = 'neural network model name')
args = parser.parse_args()

with open("config/bifnn.toml", "r") as f:
    config = toml.load(f, _dict=dict)

# === adjustable parameters ===
base_dir =       config['train_params']['base_dir'] # base directory for data
save_dir =       os.path.join(base_dir, 'outputs') # directory for saving data
epochs =           config['train_params']['epochs']        # number of epochs
batch_size =       config['train_params']['batch_size']    # batch size
learning_rate =    config['train_params']['learning_rate'] # learning rate for optimizer
step =             args.step          # step when comparing to data in x,y,z
modelout_name =    args.output_model  # file name of trained model being saved
read_model = False                  # by default, no model is read  
if (args.input_model):
    read_model =   True               # True when reading trained input model
    modelin_name = args.input_model   # name of neural network model to read
extract.ystart =   extract.ystart + args.deltaY # add offset in y for Bifrost data extraction
extract.ystop  =   extract.ystop  + args.deltaY

params = ['ux','uy','uz','lgr','lgtg'] # Bifrost parameters to read
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extract, step)
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions

input_scale, input_offset, output_scale, output_offset = scales()

# make into nx*ny*nz*nt by npar matrix with z-axis last
# Bifrost data are in [nt,nz,ny,nx] format; move z-axis from second to last/fastest position
Yorig = np.moveaxis(par,[0,1],[2,3]).reshape(-1, npar)

# coordinates of Bifrost simulation points
ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
Xorig = np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim)

# convert numpy arrays to TF tensors
X = tf.convert_to_tensor(Xorig, dtype="float32")
Y = tf.convert_to_tensor(Yorig, dtype="float32")

# loss function
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.reduce_mean(tf.square(y_true - y_pred[:,0:5]), axis=0) / tf.square(output_scale)
    )
#initialize wandb

wandb.login()
wandb_params = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "delta_y": args.deltaY,
    "output_model": modelout_name,
    "step": step,
    "scheduler": "ReduceLROnPlateau",
    "Optimizer": "Adam",
}
wandb.init(
    name=config['wandb_params']['run_name'],
    group=config['wandb_params']['group_name'],
    config=wandb_params
)

# neural network: input is (y,x,t,z), output is (ux, uy, uz, log(rho), log(T), y, x, t, z)
# all inputs and outputs are in physical units
model = dnn.model()

# if (read_model):
#     # read previous model and overwrite global model define above
#     globals()['model'] = tf.keras.models.load_model(os.path.join(save_dir, modelin_name), compile=False)

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = loss_fn
)

# save best model
check_point = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(save_dir, modelout_name),
    monitor = 'loss',
    save_best_only = True
)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

hist = model.fit(
    X,
    Y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 1,
    # callbacks=[check_point, lr_scheduler]
    callbacks=[check_point, lr_scheduler, WandbLogger()]
)

# load best model that has been saved
best_model = tf.keras.models.load_model(os.path.join(save_dir, modelout_name), compile=False)

artifact = wandb.Artifact('bifnn-model', type='model')
artifact.add_file(os.path.join(save_dir, modelout_name))
wandb.log_artifact(artifact)

# predicted solution over whole space
predicted = best_model.predict(Xorig, batch_size = nx*ny*nz//4, verbose = 0)
print('MAE data errors normalized to -1 to +1 range')
wandb_metric = {}
for i in range(npar):
    mae = np.mean(np.abs(predicted[:,i] - Yorig[:,i])) / np.asarray(output_scale)[i]
    wandb_metric[params[i]] = mae
    print(params[i], mae)
wandb.log(wandb_metric)


predicted = predicted.reshape((ny, nx, nt, nz, npar + ndim))

# coordinates for 2-D slices in neural network and Bifrost simulations
t_plot = nt // 2
z_plot = nz // 2
y_plot = ny // 2

# xy plots
for i in range(0,npar):
    plot.comparison(
        par[t_plot,z_plot,:,:,i], 
        predicted[:,:,t_plot,z_plot,i], 
        params[i],
        save=True,
        save_dir=save_dir
    )
    artifact = wandb.Artifact(f'xy_plot_{params[i]}', type='evaluation')
    artifact.add_file(os.path.join(save_dir, f'{params[i]}.png'))
    wandb.log_artifact(artifact)


# xz_plots
for i in range(0,npar):
    plot.comparison(
        par[t_plot,:,y_plot,:,i], 
        np.transpose(predicted[y_plot,:,t_plot,:,i]), 
        params[i], 
        save=True,
        save_dir=save_dir
    )
    artifact = wandb.Artifact(f'xz_plot_{params[i]}', type='evaluation')
    artifact.add_file(os.path.join(save_dir, f'{params[i]}.png'))
    wandb.log_artifact(artifact)

best_model.summary()  
wandb.finish()