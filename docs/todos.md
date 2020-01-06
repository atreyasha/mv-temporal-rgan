Workflow changes
----------------

### Model extension to MIMIC-III with conditional framework

1.  **TODO** work on extension to MIMIC-III data with
    evaluation protocols

2.  **TODO** start working on mortality data generation with
    descriptive statistics first

3.  **TODO** visualize data from MIMIC-III github repository
    in 2-dimensions to see smoothness or roughness

4.  **TODO** consider using encoder-decoder or transformers
    in GAN for variable sequence length generation

5.  **TODO** consider changing RGAN name to CRGAN
    (convolutional-recurrent-GAN), with conditional one as CRAGAN
    (auxiliary as addition)

6.  use ETH model on MIMIC-III and compare evaluations with own model

7.  apply RCGAN technique towards this process and verify results with
    existing models through TSTR/TRTS and MMD checks

8.  add custom image shapes and prepare code to shift away from square
    images

9.  read on more innovative semi-supervised gan architectures that we
    could also use

10. replace discriminator with existing supervised network to see how
    that can work better

11. before publication, publish some of the preliminary models used

### Model visualization and presentation

1.  fix column enforcements and add documentation for log files

2.  add extra option to ignore pics/gifs when cloning unless prompted

3.  add function to generate best samples from trained model aside from
    already generated images

4.  change matplotlib backend default back to instant working version
    when necessary

### Model stabilization and abstraction

1.  work on introspection tasks, where data is passed through layers
    step-wise and results are manually/automatically checked for
    explainability

2.  consider borrowing model architecture from other successful models
    and employ within local biomedical task

3.  port code to tensorflow2 for better integration -\> might solve
    problem with accuracy printing based on non-binary target labels

4.  consider that performance on images is not paramount, abstraction to
    medical data and construction of local evaluation techniques is more
    important

5.  consider developing online per-epoch similarity checks, MMD and TRTS
    to check quality of samples

6.  look up ganhacks for further possible improvements such as adding
    leaky-relu everywhere, and read on successful/innovative gan
    architectures

7.  make pipeline variable/adaptable/scalable to higher (possibly
    non-square) dimensional data in case of 64 dimensional lfw faces
    (user more variables in models instead of hard-coding)

8.  read papers for strategies/uses of synthetic data

### Heuristics

1.  add convergent pathway polynomial fit to check whether training is
    diverging or converging in given time frame

2.  add gradient checks for logging and change vis.py to include colour
    on loss line for gradients

3.  use tensorboard to analyze learning process

4.  develop similarity/quality metrics which could be used alongside
    training

### Possible architecture improvements

1.  use Wasserstein loss with standard or improved training

2.  try out vae architecture within generation process

3.  think more about constraining gradients in various network parts to
    achieve some interpretability

4.  think more about complex networks integration

5.  use feature matching and minibatch discrimination to prevent mode
    collapse

6.  consider adding Gaussian noise to images for stability (mixed
    outcomes predicted)

7.  consider resnet architecture for certain skip-connections, could be
    linked to multi-scale gradient structure

### Miscellaneous

1.  models appear more stable when discriminator is significantly less
    powerful than generator

2.  models are more stable when same noisy labels are used for
    discriminator

3.  track how many epochs or batch runs needed to converge and try to
    optimize this (\~500/2000 for mnist/lfw respectively)

4.  add MIMIC-III 2d projection depiction and learning as gif on initial
    readme

5.  remove caveats in readme once relevant developments are complete

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
