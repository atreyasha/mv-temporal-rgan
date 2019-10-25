## Multivariate recurrent GANs for generating biomedical time-series

### Overview

This project is focused on developing a recurrent GAN architecture that can imitate and generate real-time biomedical time series.

In terms of biomedical data, we will be working with the existing MIMIC-III benchmarks which are documented in [Harutyunyan, Khachatrian, Kale, Ver Steeg and Galstyan 2019](https://arxiv.org/abs/1703.07771). The MIMIC-III benchmark workflows can be found in the following public GitHub [repository](https://github.com/YerevaNN/mimic3-benchmarks).

In terms of methodologies, we are inspired by the RGAN and RCGAN architecture proposed by [Esteban, Hyland and Rätsch 2017](https://arxiv.org/abs/1706.02633). We aim to modify and further develop existing frameworks. The end goal of this project is to generate realistic biomedical time series which could enrich/mix salient medical features and possibly better encrypt/privatize biomedical data in order to inhibit retroactive patient identification.

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

### Author

Atreya Shankar, German Research Centre for Artificial Intelligence (DFKI)
