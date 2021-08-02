## Multivariate recurrent GANs

## Table of Contents

* [Motivation](#Motivation)
* [Usage](#Usage)
    * [1. Data acquisition](#1-Data-acquisition)
    * [2.i. Model training](#2i-Model-training)
    * [2.ii. Continuation of model training](#2ii-Continuation-of-model-training)
    * [3. Pruning and combination of log directories](#3-Pruning-and-combination-of-log-directories)
    * [4. Visualization](#4-Visualization)
* [Model development performance](#Model-development-performance)
* [Caveats](#Caveats)
* [Workflow changes](#Workflow-changes)
* [Acknowledgments](#Acknowledgments)

## Motivation

As mentioned in our base readme, we are motivated to generate realistic biomedical time series from the existing [MIMIC-III](https://github.com/YerevaNN/mimic3-benchmarks) benchmark dataset. In regards to our proposed GAN architecture, we are inspired by the RGAN and RCGAN (conditional RGAN) architectures proposed by [Esteban et al. 2017](https://arxiv.org/abs/1706.02633). In order to evaluate the quality of generated time series, we aim to use the evaluation techniques proposed by Esteban et al. 2017; namely utilizing Maximum Mean Discrepancy and the "Train on Synthetic, Test on Real"/"Train on Real, Test on Synthetic" frameworks.

However, we must acknowledge that a jump to these frameworks is difficult since the pipeline from data generation to evaluation is long and complex. In order to simplify this process and to establish a benchmark or proof-of-concept, we utilize a similar strategy as in Esteban et al. 2017; namely to treat existing image data as time series and to attempt to generate realistic looking images through a time series framework. One clear advantage of this technique is that generated data evaluation becomes simplified since we can (crudely) visually inspect generated images to see if they at least look realistic.

From this, we can make a simplified assumption that a recurrent GAN model that can arbitrarily generate complex images can also (probably) abstract its performance to other time series of similar dimensionality and complexity. As a result, we aim to start off by generating realistic images (modeled as time series) and to eventually extend our application to MIMIC-III biomedical data.

## Usage

### 1. Data acquisition

In order to acquire MIMIC-III data, please follow the instructions [here](https://mimic.physionet.org/gettingstarted/access/). As a note, this process involves completing an online examination on medical data ethics.

The three preliminary image datasets that we will use for early model inspection will be [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [LFWcrop Faces](https://conradsanderson.id.au/lfwcrop/). MNIST and fashion-MNIST acquisition is automated within most modern machine learning frameworks. In order to automatically acquire the greyscale version of LFWcrop faces, you can run the following script:

```shell
$ ./faces.sh
```

Upon running this script, two prompts will appear sequentially. 

1. The first prompt requests if the user wants to download and unzip the LFWcrop greyscale faces

2. The next prompt requests if the user wants to downsample the native LFWcrop greyscale faces from 64x64 to 28x28, such that it can be easier to run unit tests.

The downsampling of LFWcrop faces occurs through `pre_process_faces.py`:

```
$ python3 pre_process_faces.py --help

usage: pre_process_faces.py [-h] [--size-factor float] [--out str]

optional arguments:
  -h, --help           show this help message and exit
  --size-factor float  factor by which to upsample or downsample images (default:
                       0.4375)
  --out str            output file name (default: lfw.npy)
```

The motivation for having LFWcrop greyscale faces is that it has much fewer data instances compared to its MNIST counterparts (~13,000 vs. 60,000; similar to MIMIC-III), and faces tend to be much more complex in terms of sequential pixel patterns. As a result, we believe that effectively generating LFWcrop faces through a recurrent GAN architecture could provide a stronger backing for the ability to abstract to other complex time series such as those in the biomedical field.

### 2.i. Model training[*](#Caveats)

In order to train a RGAN/RCGAN model, you can run `train.py`. Following is the usage documentation:

```
$ python3 train.py --help

usage: train.py [-h] [--model str] [--data str] [--latent-dim int] [--epochs int]
                [--batch-size int] [--learning-rate float] [--g-factor float]
                [--droprate float] [--momentum float] [--alpha float]
                [--saving-rate int] [--continue-train] [--log-dir str]
                [--plot-model]

optional arguments:
  -h, --help            show this help message and exit
  --model str           which model to use; either RGAN or RCGAN (default: RGAN)
  --data str            which training data to use; either mnist, fashion or faces
                        (default: mnist)
  --latent-dim int      latent dimensionality of GAN generator (default: 100)
  --epochs int          number of training epochs (default: 100)
  --batch-size int      batch size for stochastic gradient descent optimization
                        (default: 256)
  --learning-rate float
                        learning rate for stochastic gradient descent optimization
                        (default: 0.0004)
  --g-factor float      factor by which generator optimizer scales discriminator
                        optimizer (default: 0.25)
  --droprate float      droprate used in GAN discriminator for
                        generalization/robustness (default: 0.25)
  --momentum float      momentum used across GAN batch-normalization (default: 0.8)
  --alpha float         alpha parameter used in leaky relu (default:
                        0.2)
  --saving-rate int     epoch period on which the model weights should be saved
                        (default: 10)
  --continue-train      option to continue training model within log directory;
                        requires --log-dir option to be defined (default: False)
  --log-dir str         log directory within ./pickles/ whose model should be
                        further trained, only required when --continue-train option
                        is specified (default: None)
  --plot-model          option to plot keras model (default: False)
```

This script will train a RGAN/RCGAN model based on the above specifications. An example of running this script is as shown:

```
$ python3 train.py --data faces --epochs 500
```

The training process will create a log directory within the `./pickles` directory, where an initialization file `init.csv`, log file `log.csv` and constant noise vector image generations will be saved. Furthermore, model weights will also be saved according to the `saving-rate` defined above. An example of a log directory name is `2019_10_20_19_02_22_RGAN_faces`, which can be simplied in the three naming subgroups: `(datetime_string)(model)(data)`.

An example tree structure of a log directory is as shown:

```shell
$ tree ./pickles/2019_10_20_19_02_22_RGAN_faces -L 1

./pickles/2019_10_20_19_02_22_RGAN_faces
├── comb_opt_weights.pickle
├── dis_opt_weights.pickle
├── dis_weights.h5
├── gen_weights.h5
├── img
├── init.csv
└── log.csv

1 directory, 7 files
```

### 2.ii. Continuation of model training[*](#Caveats)

Given the dynamic nature of model training, sometimes training procedures need to be stopped and started again at a later point in time. Our script `train.py` provides a `--continue-train` feature for doing so: 

```
$ python3 train.py --continue-train --help

usage: train.py [-h] [--model str] [--data str] [--latent-dim int] [--epochs int]
                [--batch-size int] [--learning-rate float] [--g-factor float]
                [--droprate float] [--momentum float] [--alpha float]
                [--saving-rate int] [--continue-train] --log-dir str [--plot-model]

optional arguments:
  -h, --help            show this help message and exit
  --model str           which model to use; either RGAN or RCGAN (default: RGAN)
  --data str            which training data to use; either mnist, fashion or faces
                        (default: mnist)
  --latent-dim int      latent dimensionality of GAN generator (default: 100)
  --epochs int          number of training epochs (default: 100)
  --batch-size int      batch size for stochastic gradient descent optimization
                        (default: 256)
  --learning-rate float
                        learning rate for stochastic gradient descent optimization
                        (default: 0.0004)
  --g-factor float      factor by which generator optimizer scales discriminator
                        optimizer (default: 0.25)
  --droprate float      droprate used in GAN discriminator for
                        generalization/robustness (default: 0.25)
  --momentum float      momentum used across GAN batch-normalization (default: 0.8)
  --alpha float         alpha parameter used in leaky relu (default:
                        0.2)
  --saving-rate int     epoch period on which the model weights should be saved
                        (default: 10)
  --continue-train      option to continue training model within log directory;
                        requires --log-dir option to be defined (default: False)
  --plot-model          option to plot keras model (default: False)

required name arguments:
  --log-dir str         log directory within ./pickles/ whose model should be
                        further trained, only required when --continue-train option
                        is specified (default: None)
```

Here, the log directory argument becomes a required argument. To put this more concretely, assume you already ran a model and it was saved in the following log directory `2019_10_20_19_02_22_RGAN_faces`. To continue training it, you could run the following implementation:

```
$ python3 train.py --continue-train --log-dir ./pickles/2019_10_20_19_02_22_RGAN_faces --epochs 200
```

All input features to the model (other than `--data`) can be redefined; providing the user with the ability to modify some aspects of the training evolution. All undefined input features will default to those of the previous run. Upon continuing training, a new log file with the following structure will be created: `(old_datetime_string)(model)(new_datetime_string)(data)` and the same corresponding model data will be saved here. An example of a resulting continuation log directory is `2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces`.

An example of the resulting tree structure (after continuing training) in `./pickles` would be:

```
$ tree ./pickles -L 1

./pickles
├── archive
├── 2019_10_20_19_02_22_RGAN_faces 
└── 2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces

3 directories
```

**Note:** Continuation of model training can be conducted multiple times, ie. it is possible to continue training on a log file that was already resumed:

```
$ python3 train.py --continue-train \
          --log-dir ./pickles/2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces/ --epochs 200
```

### 3. Pruning and combination of log directories

Suppose you ran multiple training sessions for a given log directory. As a result of this, you may end up having multiple sequential log directories, such as below:

```
$ tree ./pickles -L 1

./pickles
├── archive
├── 2019_10_20_19_02_22_RGAN_faces 
└── 2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces

3 directories
```

At the end of your training sessions, you can combine these directories into a single directory by using `combine_prune_logs.py`:

```
$ python3 combine_prune_logs.py --help

usage: combine_prune_logs.py [-h] --log-dir str

optional arguments:
  -h, --help     show this help message and exit

required name arguments:
  --log-dir str  base directory within pickles from which to combine/prune
                 recursively forward in time (default: None)
```

In the above defined example, you could combine both logs by running the following on the `base` or oldest directory:

```
$ python3 combine_prune_logs.py --log-dir ./pickles/2019_10_20_19_02_22_RGAN_faces 
```

This process prunes old directories and combines only the relevant results. The resulting final log directory can then be used for visualization or perhaps even further training. The final combined directory will use the newest `datetime` string in the form: `(newest_datetime_string)(model)(data)`; which would be `2019_10_24_13_45_01_RGAN_faces` in our previous example. The previous or old log directories will be moved into `./pickles/archive`, resulting in a new tree structure as below:

```
$ tree ./pickles -L 1

./pickles
├── archive
└── 2019_10_24_13_45_01_RGAN_faces

2 directories
```

**Note:** This process can also be conducted on a lone directory with no temporal descendants; in which case only that directory will be pruned for incomplete epochs and log results where the latest saved models do not apply.

### 4. Visualization

Once we have our pruned and combined log directories, we can proceed with plotting some of the training metrics with `vis.py`.

```
$ python3 vis.py --help

usage: vis.py [-h] --log-dir str [--number-ticks int] [--create-gif]
              [--shrink-factor int] [--skip-rate int] [--interval float]
              [--until int] [--progress-bar]

optional arguments:
  -h, --help           show this help message and exit
  --number-ticks int   number of x-axis ticks to use in main plots (default: 10)
  --create-gif         option to activate gif creation (default: False)
  --shrink-factor int  shrinking factor for images, applies only when --create-gif
                       is supplied (default: 4)
  --skip-rate int      skip interval when using images to construct gif applies only
                       when --create-gif is supplied (default: 2)
  --interval float     time interval when constructing gifs from images, applies
                       only when --create-gif is supplied (default: 0.1)
  --until int          set upper epoch limit for gif creation, applies only when
                       --create-gif is supplied (default: None)
  --progress-bar       option to add progress bar to gifs, appliesonly when
                       --create-gif is supplied; check readme for additional go
                       package installation instructions (default: False)

required name arguments:
  --log-dir str        base directory within pickles from which to visualize
                       (default: None)
```

This script requires a log directory or `--log-dir` as an input. It creates a subfolder `vis` in the log directory and places visualizations there. Specifically, this will create resulting loss evolution graphs and optionally a gif with a (optional) progress bar showing how constant noise vector image generations evolved with training epochs.

An example of running this script is as follows:

```
$ python3 vis.py --log-dir ./pickles/2019_10_24_13_45_01_RGAN_faces --create-gif --progress-bar
```

An example of the new tree structure of a log directory with visualizations is shown below:

```shell
$ tree ./pickles/2019_10_24_13_45_01_RGAN_faces -L 1

./pickles/2019_10_24_13_45_01_RGAN_faces
├── comb_opt_weights.pickle
├── dis_opt_weights.pickle
├── dis_weights.h5
├── gen_weights.h5
├── img
├── init.csv
├── log.csv
└── vis

2 directories, 7 files
```

## Model development performance

As our repository and models are still under development, we document various stages and performances of our models here.

### Preliminary results on non-medical data

#### RGAN

* [RGAN Model Performance Summary I](/src/docs/RGAN/RGAN_v1.md)
* [RGAN Model Performance Summary II](/src/docs/RGAN/RGAN_v2.md)
* [RGAN Model Performance Summary III](/src/docs/RGAN/RGAN_v3.md)
* [RGAN Model Performance Summary IV](/src/docs/RGAN/RGAN_v4.md)
* [RGAN Model Performance Summary V](/src/docs/RGAN/RGAN_v5.md)

#### RCGAN

* [RCGAN Model Performance Summary I](/src/docs/RCGAN/RCGAN_v1.md)
* [RCGAN Model Performance Summary II](/src/docs/RCGAN/RCGAN_v2.md)
* [RCGAN Model Performance Summary III](/src/docs/RCGAN/RCGAN_v3.md)
* [RCGAN Model Performance Summary IV](/src/docs/RCGAN/RCGAN_v4.md)

## Caveats

1. Code for model training is optimized for training on one GPU. Furthermore, the command descriptions above have omitted a flag for using a GPU, since this might be user specific. But generally, if you wish to use `CUDA-GPU '0'` you can add the local variable `CUDA_VISIBLE_DEVICES=0` as a flag to the main python script command, as shown below:

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 train.py
```

2. The models are still being tested on preliminary datasets such as MNIST; support for MIMIC-III is under development and will properly commence once performance is verified on preliminary datasets

3. RCGAN support is still under development and LFWcrop greyscale is not as yet integrated for the RCGAN

4. Model architectures are not fixed as yet and will undergo further changes in terms of increased depth and enhanced stabilization techniques

## Workflow changes

Developments to this repository and its workflow are documented in our development log [here](/docs/todos.md).

## Acknowledgments

**@eriklindernoren** Keras-GAN GitHub [repository](https://github.com/eriklindernoren/Keras-GAN) (inspired source code for this repository)
