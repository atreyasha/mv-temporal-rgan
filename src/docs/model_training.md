## i. Model training

In order to train a RGAN/RCGAN model, you can run `train.py`. Following is the usage documentation:

```
$ python3 train.py --help

usage: train.py [-h] [--data DATA] [--latent-dim LATENT_DIM] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                [--g-factor G_FACTOR] [--droprate DROPRATE]
                [--momentum MOMENTUM] [--alpha ALPHA]
                [--saving-rate SAVING_RATE] [--continue-train]
                [--log-dir LOG_DIR] [--plot-model]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           which training data to use; either mnist, fashion or
                        faces (default: mnist)
  --latent-dim LATENT_DIM
                        latent dimensionality of GAN generator (default: 100)
  --epochs EPOCHS       number of training epochs (default: 100)
  --batch-size BATCH_SIZE
                        batch size for stochastic gradient descent
                        optimization (default: 256)
  --learning-rate LEARNING_RATE
                        learning rate for stochastic gradient descent
                        optimization (default: 0.0004)
  --g-factor G_FACTOR   factor by which generator optimizer scales
                        discriminator optimizer (default: 0.25)
  --droprate DROPRATE   droprate used in GAN discriminator for
                        generalization/robustness (default: 0.25)
  --momentum MOMENTUM   momentum used across GAN batch-normalization (default:
                        0.8)
  --alpha ALPHA         alpha parameter used in discriminator leaky relu
                        (default: 0.2)
  --saving-rate SAVING_RATE
                        epoch period on which the model weights should be
                        saved (default: 10)
  --continue-train      option to continue training model within log
                        directory; requires --log-dir option to be defined
                        (default: False)
  --log-dir LOG_DIR     log directory within ./pickles/ whose model should be
                        further trained, only required when --continue-train
                        option is specified (default: None)
  --plot-model          option to plot keras model (default: False)
```

This script will train a RGAN/RCGAN model based on the above specifications. An example of running this script is as shown:

```
$ python3 train.py --data faces --epochs 500
```

The training process will create a log directory within the `./pickles` directory, where an initialization file `init.csv`, log file `log.csv` and constant noise vector image generations will be saved. Furthermore, model weights will also be saved according to the `saving-rate` defined above. An example of a log directory name is `2019_10_20_19_02_22_RGAN_faces`, which can be simplied in the three naming subgroups: `(datetime_string)(model)(data)`.

An example tree structure of a log directory is as shown:

```
$ tree -L 1

.
├── comb_opt_weights.pickle
├── comb_weights.h5
├── dis_opt_weights.pickle
├── dis_weights.h5
├── gen_weights.h5
├── img
├── init.csv
└── log.csv

1 directory, 7 files
```

## ii. Continuation of model training

Given the dynamic nature of model training, sometimes training procedures need to be stopped and started again at a later point in time. Our script `train.py` provides a `--continue-train` feature for doing so: 

```
$ python3 train.py --continue-train --help

usage: train.py [-h] [--data DATA] [--latent-dim LATENT_DIM] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                [--g-factor G_FACTOR] [--droprate DROPRATE]
                [--momentum MOMENTUM] [--alpha ALPHA]
                [--saving-rate SAVING_RATE] [--continue-train] --log-dir
                LOG_DIR [--plot-model]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           which training data to use; either mnist, fashion or
                        faces (default: mnist)
  --latent-dim LATENT_DIM
                        latent dimensionality of GAN generator (default: 100)
  --epochs EPOCHS       number of training epochs (default: 100)
  --batch-size BATCH_SIZE
                        batch size for stochastic gradient descent
                        optimization (default: 256)
  --learning-rate LEARNING_RATE
                        learning rate for stochastic gradient descent
                        optimization (default: 0.0004)
  --g-factor G_FACTOR   factor by which generator optimizer scales
                        discriminator optimizer (default: 0.25)
  --droprate DROPRATE   droprate used in GAN discriminator for
                        generalization/robustness (default: 0.25)
  --momentum MOMENTUM   momentum used across GAN batch-normalization (default:
                        0.8)
  --alpha ALPHA         alpha parameter used in discriminator leaky relu
                        (default: 0.2)
  --saving-rate SAVING_RATE
                        epoch period on which the model weights should be
                        saved (default: 10)
  --continue-train      option to continue training model within log
                        directory; requires --log-dir option to be defined
                        (default: False)
  --plot-model          option to plot keras model (default: False)

required name arguments:
  --log-dir LOG_DIR     log directory within ./pickles/ whose model should be
                        further trained, only required when --continue-train
                        option is specified (default: None)
```

Here, the log directory argument becomes a required argument. To put this more concretely, assume you already ran a model and it was saved in the following log directory `2019_10_20_19_02_22_RGAN_faces`. To continue training it, you could run the following implementation:

```
$ python3 train.py --continue-train --log-dir ./pickles/2019_10_20_19_02_22_RGAN_faces --epochs 200
```

All input features to the model (other than `--data`) can be redefined; providing the user with the ability to modify some aspects of the training evolution. All undefined input features will default to those of the previous run. Upon continuing training, a new log file with the following structure will be created: `(old_datetime_string)(model)(new_datetime_string)(data)` and the same corresponding model data will be saved here. An example of a resulting continuation log directory is `2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces`.
