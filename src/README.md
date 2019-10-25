## Multivariate recurrent GANs for generating biomedical time-series

## Table of Contents

* [Motivation](#Motivation)
* [Dependencies](#Dependencies)
* [1. Data acquisition](#1-Data-acquisition)
* [2. Model training](#2-Model-training)
* [3. Combination of log directories](#3-Combination-of-log-directories)
* [4. Visualization](#4-Visualization)

### Motivation

As mentioned in our base readme, we are motivated to generate realistic biomedical time series from the existing [MIMIC-III](https://github.com/YerevaNN/mimic3-benchmarks) benchmark dataset. In regards to our proposed GAN architecture, we are inspired by the RGAN and RCGAN architectures proposed by [Esteban et al. 2017](https://arxiv.org/abs/1706.02633). In order to evaluate the quality of generated time series, we aim to use the evaluation techniques proposed by Esteban et al. 2017; namely utilizing Maximum Mean Discrepancy and the "Train on Sythetic, Test on Real"/"Train on Real, Test on Synthetic" frameworks.

However, we must acknowledge that a jump to these frameworks is difficult since the pipeline from data generation to evaluation is long and complex. In order to simplify this process and to establish a benchmark or proof-of-concept, we utilize a similar strategy as in Esteban et al. 2017; namely to treat existing image data as time series and to attempt to generate realistic looking images through a time series framework. One clear advantage of this technique is that generated data evaluation becomes simplified since we can (crudely) visually inspect generated images to see if they at least look realistic.

From this, we can make a simplified assumption that a recurrent GAN model that can arbitrarily generate complex images can also (probably) abstract its performance to other time series of similar dimensionality and complexity. As a result, we aim to start off by generating realistic images (modeled as time series) and to eventually extend our application to MIMIC-III biomedical data.

### Dependencies

1. Install python dependencies located in `requirements.txt`:

```
$ pip install --user -r requirements.txt
```

2. Install R-based dependencies used in `gg.R`:

```R
> install.packages(c("ggplot2","tools","extrafont","reshape2","optparse","plyr"))
```

3. Optional: Install [binary](https://github.com/nwtgck/gif-progress) for adding progress bar to produced gif's.

### 1. Data acquisition

The process of acquiring MIMIC-III and other proof-of-concept datasets (MNIST, fashion-MNIST and LFWcrop faces) is documented [here](/src/docs/data_acquisition.md).

### 2. Model training

The process of training a RGAN/RCGAN model on relevant datasets, as well as resuming previously trained models is documented [here](/src/docs/model_training.md).

### 3. Combination of log directories

Training of models results in the production of log directories. The process of combining multiple log directories into a single directory is documented [here](/src/docs/combine_logs.md).

### 4. Visualization

After combining and pruning log directories, we can visualize our model results by plotting loss evolution with time and making gif's of constant noise vector projections. This procedure is documented [here](/src/docs/visualization.md)

### Acknowledgements

@eriklindernoren Keras-GAN GitHub [repository](https://github.com/eriklindernoren/Keras-GAN) (inspired source code for this repository)

<!-- ### Comments -->
<!-- * add caveat section at end with link to some areas with all exceptions due to current development (link to this in descriptions) -->
<!-- * add hook to migrate caveats from todos.org directly into relevant files -->

<!-- * provide links to model developments and stabilization techniques -->
<!-- * mention RCGAN is still under development -->
<!-- * add section for showing model results and add caveat for plotting gradients -->
<!-- * different flattening techniques, ie. as 1d time series or with more dimensions -->

<!-- * run spell-check on readme -->
