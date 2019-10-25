Workflow changes
----------------

**TODO**
====================================================================

### Model visualization and presentation

1.  **TODO** improve todos.org to todos.md pipelina and
    cavets piping

2.  **TODO** work on readme based on current results, add
    possible table of contents to readme in src for easier reading

3.  **TODO** make todos.org look better on github with proper
    dates and formatting

4.  add to readme that gif-progress/R must be installed on system

5.  make waveform type of visualization of datasets and generated
    products

6.  add MIMIC-III 2d projection depiction as gif on initial readme

7.  change matplotlib backend default back to instant working version
    when necessary

8.  add function to generate best samples from trained model aside from
    already generated image

9.  change readme information once models are developed

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

6.  **TODO** use rows as channels, separate rows from
    temporal pipeline for better convergence

7.  **TODO** work on more efficient hard model memory
    handling (saving only one instance of weights in comb.h5 and
    abstracting via layer numbers)

8.  after making above mode-based changes, run all 3 data-based models
    to see results

9.  extend models to RCGANs once results are satisfactory

10. make pipeline adaptable to higher dimensional data in case of 64
    dimensional lfw faces

### Model application to biomedical time series

1.  **TODO** visualize data from MIMIC-III github repository
    to see smoothness or roughness

2.  **TODO** apply RGAN technique towards this process and
    verify results with existing models through TSTR/TRTS and MMD checks

### Caveats

1.  add points here which will go to main readme

### Feedback from discussion

1.  try out Wasserstein loss function and dilations in CNNs

2.  convert each row to embedding

3.  pooling/attention LSTMs for higher look-back capability

4.  pass that to next LSTM row by row

5.  try extra time distribution between rows/sequences

6.  try layer vs. batch normalization vs. spectral normalization

7.  attempt SELU layer if easily available

8.  try more stable implementations of generative models such as
    R-autoencoders

9.  try architectures which do not suffer from mode collapse, eg.
    autoencoder/non-variational/transformer

### Backup Google Colab

1.  work on google drive cli access for streamlined model runs

2.  link google colab to local jupyter runtime

3.  run multiple notebooks directly from computer without browser

4.  sync google drive to localhost for easy access

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

### Clean-code/documentation

1.  make intermediate documentation and add configuration for command
    line gpu usage to readme

2.  track how many epochs or batch runs needed to converge and try to
    optimize this (\~1000 for good results)

3.  add conditions to \"train.py\" to add separate pipeline in RCGAN
    training

### Additional checks

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

    1.  use MIMIC data/models for direct TSTR/TRTS validations

    2.  use TSTR/TRTS methodologies and identification issues to
        evaluate model

    3.  combine various quality indicators to evaluate final model
        results

    4.  explore privacy perspective and whether GAN is able to remove
        personal traits

    5.  or consider another architecture which can perform this function

3.  Networks and higher-dimensions abstraction

    1.  extend to 64 pixels faces to check if abstraction possible

    2.  make model more complex to learn arbitrary sequences more
        efficiently

    3.  extend to RCGAN with realistic conditionings for actual usable
        data genration

    4.  check out mathematical proofs for convergence on GAN\'s and
        relation to Nash equilibrium

4.  Input images and feature masking

    1.  come up with mask to create or ignore feature differences

    2.  consider normalizing in a different way, via local max or
        possible integration

    3.  plot input time series as normalized 2d images to show variation

5.  Documentation and code-health:

    1.  fix unused imports and sort with python tools

    2.  encode proper documentation and model visualizations
