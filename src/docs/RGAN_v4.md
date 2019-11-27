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
<img src="/src/img/RGAN_v4/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RGAN_v4/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Gaussian-smoothened real-data labels, ie. target labels follow a normal distribution of mean `0.9` and variance `0.005` and are re-sampled every epoch
2. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer
3. Implemening spectral normalization for all significant convolutional and dense layers.

### Performance

#### i. MNIST

<p align="center">
<img src="/src/img/RGAN_v4/evolution_mnist.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN_v4/out_mnist.gif" width="650">
</p>

#### ii. Fashion-MNIST

<p align="center">
<img src="/src/img/RGAN_v4/evolution_fashion.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN_v4/out_fashion.gif" width="650">
</p>

#### iii. LFWcrop faces

<p align="center">
<img src="/src/img/RGAN_v4/evolution_faces.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN_v4/out_faces.gif" width="650">
</p>

### Next steps

These results can be deemed as satisfactory. The next step would be to extend these models to conditional ones.

