## RGAN Model Performance Summary V

### Table of Contents 

* [Generator](#Generator)
* [Discriminator](#Discriminator)
* [Stabilizing techniques](#Stabilizing-techniques)
* [Performance](#Performance)
    * [i. LFWcrop-faces](#i-LFWcrop-faces)
* [Improvements](#Improvements)

### Generator

<p align="center">
<img src="/src/img/RGAN/RGAN_v5/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RGAN/RGAN_v5/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Non-sparse activation, ie. LeakyReLU
2. Gaussian-smoothened real-data labels, ie. target labels follow a normal distribution of mean `0.9` and variance `0.005` and are re-sampled every epoch
3. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer
4. Implementing spectral normalization for all significant convolutional and dense layers.

### Performance

#### i. LFWcrop faces

<p align="center">
<img src="/src/img/RGAN/RGAN_v5/evolution_faces.png" width="800">
</p>

<p align="center">
<img src="/src/img/RGAN/RGAN_v5/out_faces.gif" width="650">
</p>

### Improvements

1. Plateau in terms of performance, no need to further test on mnist and fashion-mnist for this architecture.

2. Next step would be to extend network to conditional variant.
