## Multivariate recurrent GANs 

### Overview

This project is focused on developing recurrent GAN architecture(s) that can generate biomedical time series. In terms of methodologies, we are inspired by the RGAN and RCGAN (conditional RGAN) architecture proposed by [Esteban, Hyland and Rätsch 2017](https://arxiv.org/abs/1706.02633). In terms of biomedical data, we aim to work with the existing MIMIC-III benchmarks documented in [Harutyunyan, Khachatrian, Kale, Ver Steeg and Galstyan 2019](https://arxiv.org/abs/1703.07771) with the following public GitHub [repository](https://github.com/YerevaNN/mimic3-benchmarks).

**Note:** This project is incomplete. Details of its current status can be found [here](https://github.com/atreyasha/mv-temporal-rgan/blob/master/src/README.md#caveats).

### Dependencies

This repository's source code was tested with Python versions `3.7.*` and R versions `3.6.*`.

1. Install python dependencies located in `requirements.txt`:

    ```shell
    $ pip install -r requirements.txt
    ```

2. Install R-based dependencies:

    ```R
    > install.packages(c("ggplot2","tools","extrafont","reshape2","optparse","plyr"))
    ```

3. Install [binary](https://github.com/nwtgck/gif-progress) for adding progress bar to produced gif's.

4. **Optional:** To develop this repository, it is recommended to initialize a pre-commit hook for automatic updates of python dependencies:

    ```shell
    $ ./init.sh
    ```

### Workflow

Our workflow and source code can be found in the `src` directory of this repository. Additionally, the [readme](/src/README.md) in the `src` directory documents our functions, scripts and results.

A thorough development log for our ideas/progress can be found [here](/docs/todos.md).

### Citations

Relevant BibTeX citations for the above-mentioned papers can be found below:

Harutyunyan et al. 2019 

```
@article{Harutyunyan_2019,
   title={Multitask learning and benchmarking with clinical time series data},
   volume={6},
   ISSN={2052-4463},
   url={http://dx.doi.org/10.1038/s41597-019-0103-9},
   DOI={10.1038/s41597-019-0103-9},
   number={1},
   journal={Scientific Data},
   publisher={Springer Science and Business Media LLC},
   author={Harutyunyan, Hrayr and Khachatrian, Hrant and Kale, David C. 
   and Ver Steeg, Greg and Galstyan, Aram},
   year={2019},
   month={Jun}
}
```

Esteban et al. 2017

```
@misc{esteban2017realvalued,
    title={Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs},
    author={Cristóbal Esteban and Stephanie L. Hyland and Gunnar Rätsch},
    year={2017},
    eprint={1706.02633},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
