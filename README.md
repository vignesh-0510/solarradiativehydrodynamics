# solarradiativehydrodynamics
Modeling Solar Radiative Hydrodynamics using ResneNet and DenseNet-inspired Physics Informed Neural Networks

## Background
This project was originally based on the work from the repository:

> **[rhpinn: Radiative Hydrodynamics Physics Informed Neural Network](https://github.com/cukeller/rhpinn)**  
> Copyright © [Cristoph U. Keller]

## Requirements

- `numpy`
- `astropy`
- `matplotlib`
- `TensorFlow`


## Opacity deep neural networks

Follow the instructions in the Opacity subfolder to create the `Opacity/opacity_rosseland.keras` and `Opacity/opacity_500nm.keras` deep neural networks that output the opacity based on temperature and pressure.


## Bifrost simulation data

Please Download data as cited in the original Repository