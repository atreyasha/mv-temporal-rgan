## RCGAN Model Performance Summary II

### Table of Contents 

* [Generator](#Generator)
* [Discriminator](#Discriminator)
* [Stabilizing techniques](#Stabilizing-techniques)
* [Performance](#Performance)
    * [i. MNIST](#i-MNIST)
    * [ii. Fashion-MNIST](#ii-Fashion-MNIST)
* [Improvements](#Improvements)

### Generator

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v2/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v2/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Gaussian-smoothened real-data labels, ie. target labels follow a normal distribution of mean `0.9` and variance `0.005` and are re-sampled every epoch
2. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer
3. Implemening spectral normalization for all significant convolutional and dense layers.
4. Auxiliary architecture helps to stabilize network further

### Performance

#### i. MNIST

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v2/evolution_mnist.png" width="800">
</p>

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v2/out_mnist.gif" width="650">
</p>

#### ii. Fashion-MNIST

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v2/evolution_fashion.png" width="800">
</p>

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v2/out_fashion.gif" width="650">
</p>

### Improvements

1. Much better performance observed. General classification of images is good for both real and fake data.

2. However, quality of fake data appears to be low. We can try increasing the value of the real-data label and improving the downstream auxiliary classification architecture.
