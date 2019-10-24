## Multivariate recurrent GANs for generating biomedical time-series

### Motivation

As mentioned in our base readme, we are motivated to generate realistic biomedical time series from the existing [MIMIC-III](https://github.com/YerevaNN/mimic3-benchmarks) benchmark dataset. In regards to our proposed GAN architecture, we are inspired by the RGAN and RCGAN architectures proposed by [Esteban et al. 2017](https://arxiv.org/abs/1706.02633). In order to evaluate the quality of generated time series, we aim to use the evaluation techniques proposed by Esteban et al. 2017; namely utilizing Maximum Mean Discrepancy and the "Train on Sythetic, Test on Real"/"Train on Real, Test on Synthetic" frameworks.

However, we must acknowledge that a jump to these frameworks is difficult since the pipeline from data generation to evaluation is long and complex. In order to simplify this process and to establish a benchmark or proof-of-concept, we utilize a similar strategy as in Esteban et al. 2017; namely to treat existing image data as time series and to attempt to generate realistic looking images through a time series framework. One clear advantage of this technique is that generated data evaluation becomes simplified since we can (crudely) visually inspect generated images to see if they at least look realistic.

From this, we can make a simplified assumption that a recurrent GAN model that can arbitrarily generate complex images can also (probably) abstract its performance to other time series of similar dimensionality and complexity. As a result, we aim to start off by generating realistic images (modeled as time series) and to eventually extend our application to MIMIC-III biomedical data.

### 1. Data acquisition

In order to acquire MIMIC-III data, please follow the instructions [here](https://mimic.physionet.org/gettingstarted/access/). As a note, this process involves completing an online examination on medical data ethics.

The three preliminary image datasets that we will use for early model inspection will be [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [LFWcrop Faces](https://conradsanderson.id.au/lfwcrop/). MNIST and fashion-MNIST acquisition is automated within most modern machine learning frameworks. In order to automatically acquire the greyscale version of LFWcrop faces, you can run the following script:

```shell
$ ./init.sh
```

Upon running this script, three prompts will appear sequentially. 

1. The first requests for initiliazing a pre-commit git hook to keep python dependencies in `requirements.txt` up-to-date.

2. The next prompt asks if the user wants to download and unzip the LFWcrop greyscale faces

3. The final prompt requests if the user wants to downsample the native LFWcrop greyscale faces from 64x64 to 28x28, such that it can be easier to run unit tests.

The downsampling of LFWcrop faces occurs through `pre-process-faces.py`:

```
$ python3 pre-process-faces.py --help

usage: pre-process-faces.py [-h] [--size-factor SIZE_FACTOR] [--out OUT]

optional arguments:
  -h, --help            show this help message and exit
  --size-factor SIZE_FACTOR
                        factor by which to upsample or downsample images
                        (default: 0.4375)
  --out OUT             output file name (default: lfw.npy)
```

The motivation for having LFWcrop greyscale faces is that it has much fewer data instances compared to its MNIST counterparts (13,000 vs. 60,000; similar to MIMIC-III), and faces tend to be much more complex in terms of sequential pixel patterns. As a result, we believe that effectively generating LFWcrop faces through a recurrent GAN architecture could provide a stronger backing for the ability to abstract to other complex time series such as those in the biomedical field.

### 2.i. Model training

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

### 2.ii. Continuation of model training

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

All input features to the model (other than `--data`) can be redefined; providing the user with the ability to modify some aspects of the training evolution. Upon continuing training, a new log file with the following structure will be created: `(old_datetime_string)(model)(new_datetime_string)(data)` and the same corresponding model data will be saved here. An example of a resulting continuation log directory is `2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces`.

### Comments

* show utility of combine logs
* show visualizations
* add TOC to long readme
* add caveat section at end with link to some areas with all exceptions due to current development
* mention RCGAN is still under development

* provide links to model developments and stabilization techniques
* link actual model performances and descriptions below
* different flattening techniques, ie. as 1d time series or with more dimensions
* section on current performance of models and next steps based on performances

* add acknowledgements for keras GAN implementations
* run spell-check on readme
