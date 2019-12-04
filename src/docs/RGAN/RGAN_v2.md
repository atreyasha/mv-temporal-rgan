## RGAN Model Performance Summary II

### Table of Contents 

* [Generator](#Generator)
* [Discriminator](#Discriminator)
* [Stabilizing techniques](#Stabilizing-techniques)
* [Performance](#Performance)
    * [i. MNIST](#i-MNIST)
    * [ii. Fashion-MNIST](#ii-Fashion-MNIST)
    * [iii. LFWcrop-faces](#iii-LFWcrop-faces)
* [Improvements](#Improvements)

### Generator

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Smoothened labels for GAN discriminator, ie. target label value of 0.9 instead of 1 for identifying real data
2. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer

### Performance

#### i. MNIST

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/evolution_mnist.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/out_mnist.gif" width="650">
</p>

#### ii. Fashion-MNIST

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/evolution_fashion.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/out_fashion.gif" width="650">
</p>

#### iii. LFWcrop faces

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/evolution_faces.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v2/out_faces.gif" width="650">
</p>

### Improvements
1. We implemented 2d convolutions in this model run instead of 1d convolutions compared to the previous runs. This led to some improvements, especially with MNIST. However, we can see that passing multivariate data into LSTM's leads to jagged lines and a lack of general details. We could possibly combine 2d convolutions with LSTM's that read data as 1d-strings.

2. We could implement more stabilization techniques such as noise addition, spectral normalization and multi-scale gradients.

### Feedback

We implemented Spectral Normalization but it showed minimal improvements. We will try again to implement it with the next testing model.
