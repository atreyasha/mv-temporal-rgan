## RGAN Model Performance Summary III

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
<img src="/src/img/RGAN/RGAN_v3/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Gaussian-smoothened real-data labels, ie. target labels follow a normal distribution of mean `0.9` and variance `0.005` and are re-sampled every epoch
2. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer

### Performance

#### i. MNIST

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/evolution_mnist.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/out_mnist.gif" width="650">
</p>

#### ii. Fashion-MNIST

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/evolution_fashion.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/out_fashion.gif" width="650">
</p>

#### iii. LFWcrop faces

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/evolution_faces.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v3/out_faces.gif" width="650">
</p>

### Improvements
1. We mixed 2d convolutions with 1d lstm string techniques and can observe more fine-grained patterns in the GAN generation. The only issue is a systemic value of the final row in the images being bright for mnist and dark for faces. This will be fixed in the next model development by using a bidirectional lstm in the discriminator.

2. We could implement more stabilization techniques such as noise addition, spectral normalization and multi-scale gradients.
