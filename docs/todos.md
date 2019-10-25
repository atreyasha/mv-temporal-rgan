Workflow changes
----------------

### Model visualization and presentation

1.  **TODO** add initial model
    summary/architectures/feedback/performance into readme

2.  work on better connection between readme and development-log by
    piping certain points on pre-commit hooks

3.  make todos.org look better on github with proper dates and
    formatting

4.  make waveform type of visualization of datasets and generated
    products for preliminary datasets and MIMIC-III

5.  add function to generate best samples from trained model aside from
    already generated image

6.  change matplotlib backend default back to instant working version
    when necessary

7.  add MIMIC-III 2d projection depiction and learning as gif on initial
    readme

### Model stabilization and abstraction

1.  **TODO** use Gaussian noise addition to images (to
    prevent mode collapse and add robustness) and Gaussian-noise label
    smoothing (non-model)

2.  **TODO** use multi-scale gradient and spectral
    normalization (model), first non-model then model-based changes

3.  **TODO** make network deeper and review eth rgan model
    for comparison

4.  **TODO** use Wasserstein loss and dilations for larger
    scale temporal relations

5.  **TODO** consider removing LSTM in generator and adding
    additional LSTM in discriminator

6.  **TODO** use rows as channels/time-steps, separate rows
    from temporal pipeline for better convergence

7.  **TODO** work on more efficient hard model memory
    handling (saving only one instance of weights in comb.h5 and
    abstracting via layer numbers)

8.  after making above mode-based changes, run all 3 data-based models
    to see results

9.  extend models to RCGANs once results are satisfactory

10. make pipeline variable/adaptable/scalable to higher dimensional data
    in case of 64 dimensional lfw faces

### Model application to biomedical time series

1.  visualize data from MIMIC-III github repository in 2-dimensions to
    see smoothness or roughness

2.  apply RCGAN technique towards this process and verify results with
    existing models through TSTR/TRTS and MMD checks

### Feedback from discussions

1.  pooling/attention LSTMs for higher look-back capability

2.  extra time distribution between rows/sequences

3.  layer vs. batch normalization

4.  attempt SELU layer if easily available

5.  more stable implementations of generative models such as
    R-autoencoders

6.  architectures which do not suffer from mode collapse, eg.
    autoencoder/non-variational/transformer

### GPU management

1.  try to share memory and tasks between gpus (multi~gpumodel~)

2.  try to use memory limiting in code in case GPUs are already being
    used

### Heuristics

1.  add gradient checks for logging and change vis.py to include colour
    on loss line for gradients

2.  set up crowd-sourced grid-search via emails to check results

3.  optionally use tensorboard to analyze learning process

4.  look for similarity measure metrics which could be used alongside
    training

### Backup Google Colab

1.  work on google drive cli access for streamlined model runs

2.  link google colab to local jupyter runtime

3.  run multiple notebooks directly from computer without browser

4.  sync google drive to localhost for easy access

### Clean-code/documentation

1.  track how many epochs or batch runs needed to converge and try to
    optimize this (\~1000 for good results)

2.  add conditions to \"train.py\" to add separate pipeline in RCGAN
    training

### Additional improvements

1.  look into unsupervised feature extraction in ML

2.  isolate personal identification features in discriminator from
    generated time series

3.  use adversarial samples to generate bad data that network falsely
    predicts

### Brainstorming points

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

    2.  make proper documentation and model visualizations
