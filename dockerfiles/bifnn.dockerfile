FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

RUN pip install glob2  wandb seaborn astropy numpy matplotlib

RUN pip install tensorflow-addons toml

RUN mkdir /app

WORKDIR /app
