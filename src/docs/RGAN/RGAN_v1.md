## RGAN Model Performance Summary I

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
<img src="/src/img/RGAN/RGAN_v1/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Smoothened labels for GAN discriminator, ie. target label value of 0.9 instead of 1 for identifying real data
2. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer

### Performance

#### i. MNIST

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/evolution_mnist.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/out_mnist.gif" width="650">
</p>

#### ii. Fashion-MNIST

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/evolution_fashion.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/out_fashion.gif" width="650">
</p>

#### iii. LFWcrop faces

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/evolution_faces.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v1/out_faces.gif" width="650">
</p>

### Improvements
1. We can observe incomplete convergence in the case of MNIST and LFWcrop faces. This could be due to mode collapse. We would need to incorporate some strategies such as noise introduction to overcome mode collapse.

2. We could implement more stabilization techniques such as noise addition, spectral normalization and multi-scale gradients.

3. Currently, the images are modeled as 1d strings or vectors of length 784. We could attempt to model them as smaller vectors with channels or depth, such that each time step will be encoded as a multi-dimensional event instead of diffuse in a 1d string situation.
