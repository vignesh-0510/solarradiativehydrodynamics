# train network to make initial guesses of physical parameters uz, lr, lt
# based on normalized continuum intensity
#   - uses Bifrost simulations directly and not a DNN trained on Bifrost
#   - applies a weight in the z-axis to maximize correspondence in layers where light emerges from
# additional constraints:
#   - zero net mass motion through each horizontal layer
#   - mean vz^2 compatible with zero net mass flux through layers
#   - mean density and temperature stratifications
#   - variance of the density and temperature
#   - correlation between mass flux and temperature
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from readBifrost import readBifrost
from scales import scales
import numpy as np
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import tensorflow as tf
import os
import toml
import wandb
from utils import WandbLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models.int2mod import model

# parse arguments
parser = argparse.ArgumentParser(
    prog = 'int2mod',
    description = 'Train a neural network to provide physical parameters based on intensity images alone'
)
parser.add_argument('-y', '--deltaY',
                    type = int,
                    default = 0,
                    help = 'change in y extraction')
parser.add_argument('-i', '--image_directory',
                    type = str,
                    help = 'image directory')
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'int2mod.keras',
                    help = 'neural network output model name')
args = parser.parse_args()

with open("config/int2mod.toml", "r") as f:
    config = toml.load(f, _dict=dict)

# === adjustable parameters ===
base_dir =       config['train_params']['base_dir'] # base directory for data
save_dir =       os.path.join(base_dir, 'outputs') # directory for saving data
image_dir =       os.path.join(base_dir, 'images') # directory for saving data
epochs =           config['train_params']['epochs']        # number of epochs
batch_size =       config['train_params']['batch_size']    # batch size
learning_rate =    config['train_params']['learning_rate'] # learning rate for optimizer
model_name =       args.output_model  # file name of trained model being saved
extr.ystart =      extr.ystart + args.deltaY # add offset in y for Bifrost data extraction
extr.ystop  =      extr.ystop  + args.deltaY
image_name =       os.path.join(image_dir, args.image_directory)     # directory with images

params = ['uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, 1)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
input_scale, input_offset, output_scale, output_offset = scales()

# weight in z-axis for fitting BiFrost parameters, more weight where light emerges
znorm = zaxis*input_scale[3] + input_offset[3]
zweight = np.sqrt(1.0 + np.exp(-np.power(znorm / 0.25, 2)) * 20.0)
zweight = tf.convert_to_tensor(zweight / np.mean(zweight), dtype="float32") # normalize to a mean of 1.0

# remove net mass flux from BiFrost simulations
par = np.moveaxis(par,1,3) # moves z-axis to last coordinate place
for it in range(nt):
    dens = np.power(10.0, par[it,...,1])
    par[it,...,0] = par[it,...,0] - (
        np.mean(par[it,...,0]*dens, axis=(0,1)) / np.mean(dens, axis=(0,1))
    )

# load intensity images and normalize to mean 0 and stdv 1
imag = np.zeros((nt,ny,nx), dtype=np.float32)
for it in range(nt):
    imag[it,:,:] = fits.getdata(image_name+'/image'+str(it)+'.fits')
    imag[it,:,:] = imag[it,:,:] - np.mean(imag[it,:,:])
    imag[it,:,:] = imag[it,:,:] / np.std(imag[it,:,:])
imag = tf.convert_to_tensor(imag.flatten(), dtype="float32")

ndim = 2 # number of dimensions: (flattened) intensity, z
im, zm = np.meshgrid(imag, zaxis, indexing='ij')
X_orig = np.stack((im, zm), axis = -1).reshape(-1, nz, ndim)
Y_orig = par.reshape(-1, nz, npar)

# variance of vertical velocity
vvz =   np.var( Y_orig[:,:,0], axis=0)
# mean and variance of log density
mlrho = np.mean(Y_orig[:,:,1], axis=0)
vlrho = np.var( Y_orig[:,:,1], axis=0)
# mean and variance of log temperature
mlogt = np.mean(Y_orig[:,:,2], axis=0)
vlogt = np.var( Y_orig[:,:,2], axis=0)

# correlation between temperature and mass flux
vz = Y_orig[:,:,0]
rho = np.power(10, Y_orig[:,:,1])
# --->>> this should probably be done per time step and not averaged over time!
massflux = rho * vz - np.mean(rho * vz, axis=0)
temp = np.power(10.0, Y_orig[:,:,2])
vtcorr = np.mean(
    (temp - np.mean(temp, axis=0, keepdims = True)) * massflux, 
    axis=0
)

# Keras custom sequence generator that shuffles t,x,y points for every epoch
class CustomSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        super().__init__()
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        # print('len(self.x)',len(self.x))
        p = np.random.permutation(len(self.x))
        self.x_shuffled = self.x[p,:,:].reshape((-1,ndim))
        self.y_shuffled = self.y[p,:,:].reshape((-1,npar))

    # returns number of batches
    def __len__(self):
        return int(np.ceil(len(self.x_shuffled) / float(self.batch_size)))

    # returns one batch at batch index idx
    def __getitem__(self, idx):
        batch_x = self.x_shuffled[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_shuffled[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    # reshuffle at end of each epoch   
    def on_epoch_end(self):
        p = np.random.permutation(len(self.x))
        self.x_shuffled = self.x[p,:,:].reshape((-1,ndim))
        self.y_shuffled = self.y[p,:,:].reshape((-1,npar))


# neural network:
#   input is normalized intensity and z-coordinate in meters
#   output is uz, log10(rho), log10(T) along z
#   all inputs and outputs are in physical coordinates
# model = tf.keras.Sequential([
#     tf.keras.layers.Input((ndim,)),
#     tf.keras.layers.Rescaling(
#         scale  = [1.0, input_scale[3]],
#         offset = [0.0, input_offset[3]]
#     ),
#     tf.keras.layers.Dense(8, activation='tanh'),
#     tf.keras.layers.Dense(16, activation='tanh'),
#     tf.keras.layers.Dense(32, activation='tanh'),
#     tf.keras.layers.Dense(npar, activation='linear'),
#     tf.keras.layers.Rescaling(
#         # [2:5] is scaling for [uz, log10(rho), log10(T)]
#         scale  = output_scale[2:5],
#         offset = output_offset[2:5]
#     )
# ])
model = model(ndim, npar)


def losses(y_true, y_pred):
    # minimizing vertical net mass flow should only control vertical velocity
    rho = tf.reshape(tf.pow(10.0, tf.stop_gradient(y_pred[:,1])),[-1,nz])
    vz = tf.reshape(y_pred[:,0],[-1,nz])
    # --->>> this should probably be done per time step and not averaged over time!
    massflux = rho * vz - tf.reduce_mean(rho * vz, axis=0, keepdims = True)
    temp = tf.reshape(tf.pow(10.0, y_pred[:,2]),[-1,nz])

    return tf.stack([
        # mean vertical mass motion in each layer should be zero
        # --->>> should be true for each time step
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(tf.reshape(y_pred[:,0], [-1,nz]) * rho, axis=0) /
                tf.reduce_mean(rho, axis=0)
            )
        ),

        # difference of variance of vertical velocity
        tf.reduce_mean(
            tf.square(
                tf.math.reduce_variance(tf.reshape(y_pred[:,0], [-1,nz]), axis=0) - vvz
            )
        ),

        # difference of mean of logarithm of density
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(tf.reshape(y_pred[:,1], [-1,nz]), axis=0) - mlrho
            )
        ),

        # difference of variance of logarithm of density
        tf.reduce_mean(
            tf.square(
                tf.math.reduce_variance(tf.reshape(y_pred[:,1], [-1,nz]), axis=0) - vlrho
            )
        ),

        # difference of mean of logarithm of temperature
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(tf.reshape(y_pred[:,2], [-1,nz]), axis=0) - mlogt
            )
        ),

        # difference of variance of logarithm of temperature
        tf.reduce_mean(
            tf.square(
                tf.math.reduce_variance(tf.reshape(y_pred[:,2], [-1,nz]), axis=0) - vlogt
            )
        ),

        # correlation between mass flux and temperature         
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(
            (temp - tf.reduce_mean(temp, axis=0, keepdims = True)) * massflux, 
            axis=0) - vtcorr
            )
        ),        

        # difference of vz
        tf.reduce_mean(
            tf.reshape(tf.square(y_true[:,0] - y_pred[:,0]),[-1,nz]) * zweight) / tf.square(output_scale[2]),

        # difference of log10(rho)
        tf.reduce_mean(
            tf.reshape(tf.square(y_true[:,1] - y_pred[:,1]),[-1,nz]) * zweight) / tf.square(output_scale[3]),

        # difference of log10(T)
        tf.reduce_mean(
            tf.reshape(tf.square(y_true[:,2] - y_pred[:,2]),[-1,nz]) * zweight) / tf.square(output_scale[4]),
    ])

loss_name = [
    'mean vertical mass motion           ',
    'variance of vertical velocity       ',
    'mean of logarithm of density        ',
    'variance of logarithm of density    ',
    'mean of logarithm of temperature    ',
    'variance of logarithm of temperature',
    'mass flux temperature correlation   ',
    'vz                                  ',
    'log10(rho)                          ',
    'log10(T)                            ',
]

loss_scale = [
    1e-6,  # mean vertical mass motion
    1e-13, # variance of vertical velocity
    1e-1,  # mean of logarithm of density
    1e1,   # variance of logarithm of density
    1e1,   # mean of logarithm of temperature
    1e2,   # variance of logarithm of temperature
    1e-5,  # mass flux temperature correlation
    1,     # vz
    1,     # log10(rho)
    1,     # log10(T)
]

def loss_fn(y_true, y_pred):
    return tf.reduce_sum(losses(y_true, y_pred) * loss_scale)

wandb.login()
wandb_params = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "delta_y": args.deltaY,
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

# save best model
check_point = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(save_dir, model_name),
    monitor = 'loss',
    save_best_only = True
)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

seq = CustomSequence(X_orig, Y_orig, batch_size)

model.fit(
    seq,
    epochs = epochs,
    verbose = 1,
    shuffle = False,
    callbacks=[check_point, lr_scheduler, WandbLogger()]
)

# print neural network structure
model.summary()

# load best model
best_model = tf.keras.models.load_model(os.path.join(save_dir, model_name), compile=False)
artifact = wandb.Artifact('int2mod-model', type='model')
artifact.add_file(os.path.join(save_dir, model_name))
wandb.log_artifact(artifact)

pred = best_model.predict(X_orig.reshape(-1,ndim), batch_size=nx*nz*8).reshape((-1,nz,npar))
print(' ')
print('Mean Absolute Errors (from BiFrost) normalized to -1 to +1 range')
wandb_metric = {}
for i in range(len(params)):
    mae = np.mean(np.abs(Y_orig[:,:,i] - pred[:,:,i]) / output_scale[i+2])
    print(params[i], mae)
    wandb_metric[params[i]] = mae

print('losses')
l = losses(Y_orig.reshape(-1,npar), model(X_orig.reshape(-1,ndim)))
for i in range(len(l)):
    loss_val = float(l[i] * loss_scale[i])
    wandb_metric[loss_name[i]] = loss_val
    print(loss_name[i], float(l[i]*loss_scale[i]))

wandb.log(wandb_metric)

p = pred
d = Y_orig

for i in range(len(params)):
    plt.plot(zaxis/1e3, np.mean(np.abs(d[:,:,i] - p[:,:,i]), axis=0) / output_scale[i])
    plt.title('MAE of ' + params[i])
    plt.xlabel('z[km]')
    plt.ylabel('MAE')
    plt.savefig(os.path.join(image_dir, 'int2mod', f'mae_{params[i]}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    artifact = wandb.Artifact(f'mae_{params[i]}.png', type='evaluation')
    artifact.add_file(os.path.join(image_dir, 'int2mod', f'mae_{params[i]}.png'))
    wandb.log_artifact(artifact)

plt.plot(zaxis/1e3, 
    np.mean(d[:,:,0] * np.power(10,d[:,:,1]), axis=0) / 
    np.mean(np.power(10,d[:,:,1]), axis=0),
    label="BiFrost mass flux velocity"
)
plt.plot(zaxis/1e3, 
    np.mean(p[:,:,0] * np.power(10,p[:,:,1]), axis=0) / 
    np.mean(np.power(10,p[:,:,1]), axis=0),
    label="Fit mass flux velocity"
)
plt.title('mean vertical mass flux velocity')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('vz[m/s]')
plt.legend()
plt.savefig(os.path.join(image_dir, 'int2mod', 'mean_vertical_mass_flux_velocity.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'mean_vertical_mass_flux_velocity.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'mean_vertical_mass_flux_velocity.png'))
wandb.log_artifact(artifact)
plt.show()

plt.plot(zaxis/1e3, vvz, label="BiFrost VarVz")
plt.plot(zaxis/1e3, np.var(p[:,:,0], axis=0), label="Fit VarVz")
plt.title('variance of vertical velocity')
plt.xlabel('z[km]')
plt.ylabel('(vz[m/s])^2')
plt.legend()
plt.savefig(os.path.join(image_dir, 'int2mod', 'variance_vertical_velocity.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'variance_vertical_velocity.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'variance_vertical_velocity.png'))
wandb.log_artifact(artifact)
plt.show()

plt.plot(zaxis/1e3, mlrho, label="BiFrost MeanLogRho")
plt.plot(zaxis/1e3, np.mean(p[:,:,1], axis=0), label="Fit MeanLogRho")
plt.title('mean log(density)')
plt.xlabel('z[km]')
plt.ylabel('rho[kg/m^3]')
plt.legend()
plt.savefig(os.path.join(image_dir, 'int2mod', 'mean_log_density.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'mean_log_density.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'mean_log_density.png'))
wandb.log_artifact(artifact)
plt.show()

plt.plot(zaxis/1e3, vlrho, label="BiFrost VLogRho")
plt.plot(zaxis/1e3, np.var(p[:,:,1], axis=0), label="Fit VLogRho")
plt.title('variance log(density)')
plt.xlabel('z[km]')
plt.ylabel('rho[kg/m^3]^2')
plt.legend()
plt.savefig(os.path.join(image_dir, 'int2mod', 'variance_log_density.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'variance_log_density.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'variance_log_density.png'))
wandb.log_artifact(artifact)
plt.show()

plt.plot(zaxis/1e3, mlogt, label="BiFrost MeanLogT")
plt.plot(zaxis/1e3, np.mean(p[:,:,2], axis=0), label="Fit MeanlogT")
plt.title('mean log(temperature)')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('T[K]')
plt.savefig(os.path.join(image_dir, 'int2mod', 'mean_log_temperature.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'mean_log_temperature.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'mean_log_temperature.png'))
wandb.log_artifact(artifact)
plt.show()

plt.plot(zaxis/1e3, vlogt, label="BiFrost VarLogT")
plt.plot(zaxis/1e3, np.var(p[:,:,2], axis=0), label="Fit VarLogT")
plt.title('variance log(temperature)')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('T[K]^2')
plt.savefig(os.path.join(image_dir, 'int2mod', 'variance_log_temperature.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'variance_log_temperature.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'variance_log_temperature.png'))
wandb.log_artifact(artifact)
plt.show()

# correlation between temperature and mass flux as a function of height
temp = np.power(10.0, p[:,:,2])
mf = p[:,:,0] * np.power(10.0,p[:,:,1]) - np.mean(p[:,:,0] * np.power(10.0,p[:,:,1]), axis=0)
mtcorr = np.mean(
    (temp - np.mean(temp, axis=0, keepdims = True)) * mf, 
    axis=0
)
plt.plot(zaxis/1e3, vtcorr, label="BiFrost VTCorr")
plt.plot(zaxis/1e3, mtcorr, label="Fit MTCorr")
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('vz*rho-T correlation')
plt.title('vertical mass flux - temperature correlation')
plt.savefig(os.path.join(image_dir, 'int2mod', 'vz_rho_temp_correlation.png'), dpi=300, bbox_inches='tight')
artifact = wandb.Artifact(f'vz_rho_temp_correlation.png', type='evaluation')
artifact.add_file(os.path.join(image_dir, 'int2mod', 'vz_rho_temp_correlation.png'))
wandb.log_artifact(artifact)
plt.show()

wandb.finish()