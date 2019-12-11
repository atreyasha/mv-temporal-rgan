## RCGAN Model Performance Summary III

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
<img src="/src/img/RCGAN/RCGAN_v3/gen.png" width="300">
</p>

### Discriminator

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v3/dis.png" width="300">
</p>

### Stabilizing Techniques

1. Non-sparse activation, ie. LeakyReLU
2. Implementing two optimizers in GAN with differing learning rates; specifically where the generator has slightly lower learning rate than optimizer
3. Implemening spectral normalization for all significant convolutional and dense layers.
4. Auxiliary architecture helps to stabilize network further

### Performance

#### i. MNIST

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v3/evolution_mnist.png" width="800">
</p>

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v3/out_mnist.gif" width="650">
</p>

#### ii. Fashion-MNIST

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v3/evolution_fashion.png" width="800">
</p>

<p align="center">
<img src="/src/img/RCGAN/RCGAN_v3/out_fashion.gif" width="650">
</p>

### Improvements

1. Improved quality of images produced.

2. Reduced classification accuracy, some labels are misclassified.

3. Need to find a middle-ground between generated image quality and generated label accuracy. This needs further adjustment.
