# EchoCemDataChallenge
Evaluation of the ciment quality of gas well casings through segmentation of ultrasonic images


## Technical description

This repository contains the code for the evaluation of the ciment quality of gas well casings through segmentation of ultrasonic images.

The code is written in Python and uses the hydra, pytorch and pytorch lightning libraries. It is organized in a modular way, with each module being responsible for a specific task.

The code is organized in the following way:

```python
.
├── data/
│   ├── raw/
│   │   ├── X_train/
│   │   │   ├── images/
│   │   │   │   ├── well_{n}_section_{m}_path_{i}.npy
│   │   │   │   └── ...
│   │   │   └── patch_annotations/ # annotations extracted for Y_train.csv
│   │   │       └── well_{n}_section_{m}_path_{i}.npy
│   │   ├── X_test/
│   │   │   └── images/
│   │   │       └── ...
│   │   ├── X_unlabeled/
│   │   │   └── images/
│   │   │       └── ...
│   │   └── Y_train.csv
│   └── processed/
│       ├── X_train/
│       │   ├── images/
│       │   │   └── {n}_{m}.npy # raw patches aggregated into huge images
│       │   └── annotations/
│       │       └── {n}_{m}.npy
│       └── ...
├── models/ # saved models
├── notebooks/ # notebooks to manipulate the different modules / test and debug
└── src/echocem/
    ├── callback/ # callbacks during training
    ├── conf/     # hydra config files
    │   ├── trainer.yaml # pytorch-lightning trainer
    │   ├── model.yaml   # model / losses / ...
    │   └── data.yaml    # train & val dataloaders
    ├── data/     # data loading
    │   ├── data.py
    │   ├── utils.py
    │   └── data_transforms.py
    └── models/
        ├── evaluation.py
        ├── layers.py
        ├── losses.py
        ├── models.py
        └── trainer.py
```

Once everything is set up, training a simple model is as easy as:
```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# load the configuration file for the data
data_cfg = OmegaConf.load("src/echocem/conf/data.yaml")
train_loader = instantiate(data_cfg.train_dataloader)
val_loader = instantiate(data_cfg.validation_dataloader)
# load the configuration file for the model
model_cfg = OmegaConf.load("src/echocem/conf/model.yaml")
segmModel = instantiate(model_cfg.segmModel)
# load the configuration file for the trainer
trainer_cfg = OmegaConf.load("src/echocem/conf/trainer.yaml")
trainer = instantiate(trainer_cfg.trainer)
# train the model
trainer.fit(segmModel, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

Every parameter can be changed in the configuration files. The configuration files are yaml file that contains the parameters of the model, the data, the trainer, etc.
The code supports an arbitrary number of metrics to be computed and displayed during training. Thanks to pytorch lightning, the code is modular and can be easily extended, and the training can be easily parallelized on multiple GPUs.

Every module is documented with docstrings and the code is written in a modular way. Each module is tested.