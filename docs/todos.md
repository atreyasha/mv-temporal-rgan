Workflow changes
----------------

### Model extension to RCGAN

1.  **TODO** upate all pipelines to account for RCGAN, save
    obj logs correctly and make nice documentation

2.  **TODO** modify continue-train for rcgan, add custom
    image shapes and prepare code to shift away from square images

3.  **TODO** make plotting procedure more efficient within
    rcgan

4.  **TODO** check if additional transformation or np.reshape
    for labels is required, read on more innovative gan architectures
    that we could also use

5.  **TODO** look up for normalizing feature in terms of
    input data, perhaps this might help (or revert back)

6.  **TODO** if normalizing feature helps, perhaps re-run
    newly normalized models with basic RGAN

7.  **TODO** add rcgan to plot model functionality

8.  **TODO** modify lfw for rcgan with artificially generated
    labels from other classifiers

9.  publish separate annotated lfw-crop dataset, or consider using
    CIFAR-10 if small dataset like MIMIC-III

10. bottom horizontal artifact keeps showing, investigate origin and how
    to circumvent

11. update readme with relevant changes to RCGAN

12. make consistent dataset between RGAN and RCGAN, re-run on RGAN after
    re-normalizing, re-run face code

13. before publication, run models on simple data to get best results
    and publish models used (need to decide on normlization technique)

### Model stabilization and abstraction

1.  **TODO** work on more efficient (automated) hard model
    memory handling (saving only one instance of weights in comb.h5 and
    abstracting via layer numbers) -\> necessary for github clones to be
    light

2.  **TODO** export optimizer weights as h5 instead of pickle
    for data consistency and compactness

3.  **TODO** make modular function for model restoration,
    edit readme with new continue training details afterwards

4.  **TODO** consider borrowing model architecture from other
    successful models and employ within local biomedical task

5.  work on introspection tasks, where data is passed through layers
    step-wise and results are manually/automatically checked for
    explainability

6.  port code to tensorflow2 for better integration -\> might solve
    problem with accuracy printing based on non-binary target labels

7.  consider that performance on images is not paramount, abstraction to
    medical data and construction of local evaluation techniques is more
    important

8.  consider developing online per-epoch similarity checks, MMD and TRTS
    to check quality of samples

9.  look up ganhacks for further possible improvements such as adding
    leaky-relu everywhere, and read on successful/innovative gan
    architectures

10. make pipeline variable/adaptable/scalable to higher (possibly
    non-square) dimensional data in case of 64 dimensional lfw faces
    (user more variables in models instead of hard-coding)

11. read papers for strategies/uses of synthetic data

### Model visualization and presentation

1.  **TODO** add extra option to ignore pics/gifs when
    cloning unless prompted

2.  add function to generate best samples from trained model aside from
    already generated images

3.  change matplotlib backend default back to instant working version
    when necessary

### Model extension to biomedical time series

1.  visualize data from MIMIC-III github repository in 2-dimensions to
    see smoothness or roughness

2.  use ETH model on MIMIC-III and compare evaluations with own model

3.  apply RCGAN technique towards this process and verify results with
    existing models through TSTR/TRTS and MMD checks

### Heuristics

1.  **TODO** add convergent pathway polynomial fit to check
    whether training is diverging or converging in given time frame

2.  add gradient checks for logging and change vis.py to include colour
    on loss line for gradients

3.  use tensorboard to analyze learning process

4.  develop similarity/quality metrics which could be used alongside
    training

### Possible architecture improvements

1.  use Wasserstein loss with standard or improved training

2.  use feature matching and minibatch discrimination to prevent mode
    collapse

3.  consider adding Gaussian noise to images for stability (mixed
    outcomes predicted)

4.  consider resnet architecture for certain skip-connections, could be
    linked to multi-scale gradient structure

### Miscellaneous

1.  models appear more stable when discriminator is significantly less
    powerful than generator

2.  models are better when same noisy labels are used for both generator
    and discriminator

3.  track how many epochs or batch runs needed to converge and try to
    optimize this (\~500/2000 for mnist/lfw respectively)

4.  add conditions to \"train.py\" to add separate pipeline in RCGAN
    training

5.  add MIMIC-III 2d projection depiction and learning as gif on initial
    readme

6.  remove caveats in readme once relevant developments are complete

### High-level ideas

1.  GAN stabilisation:

    1.  Gaussian label smoothing

    2.  differing learning rates for optimizers

    3.  Gaussian noise addition to images

    4.  spectral normalization

    5.  multi-scale gradient

2.  Evaluation pipeline

    1.  use MIMIC data/models for direct MMD + TSTR/TRTS validations

    2.  explore privacy perspective and whether GAN is able to remove
        personal traits

    3.  or consider another architecture which can perform this function

3.  Networks and higher-dimensions abstraction

    1.  extend to deeper model which can handle 64 pixels faces to check
        if abstraction possible

    2.  extend to RCGAN with realistic conditionings for actual usable
        data genration

4.  Input images and feature masking

    1.  come up with mask to simulate missing data in real-life

    2.  compare input and output images as time series with signals

5.  Documentation and code-health:

    1.  fix unused imports and sort with python tools

    2.  make detailed documentation and model visualizations
