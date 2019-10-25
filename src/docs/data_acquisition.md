## Data Acquisition

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
