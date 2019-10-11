## Workflow to-do's

### Important
* configure code to use specific gpus on cluster
* make function to continue training within train main file
* make pipeline adaptable to image pixels or modify dimensionality of lfw-faces

### Relevant concepts
* consider return sequences vs. return neurons
* make encoding and decoding lstm with cnn to compact architecture
* consider using stacked LSTMs

### Heuristics
* set up crowd-sourced grid-search via emails to check results
* save optimizer to re-use when loading saved model
* add gradient checks and heuristics for early stopping mechanism
* add grid-search mechanism for checking more possibilities
* make mechanism for dynamic g-factor adjustment
* optionally use tensorboard to analyze learning process

### Extra steps
* try to share memory and tasks between gpus (multi\_gpu\_model)
* take into account memory of system before running
* take into account gpu usage before executing

### Brainstorming points

#### Evaluation pipeline
* train mnist, fashion-mnist and lfw-faces for 28 pixels
* extend to 64 pixels faces to check if abstraction possible
* use TSTR/TRTS methodologies and identification issues to evaluate model
* use MIMIC data/models for direct TSTR/TRTS validations
* explore privacy perspective and whether GAN is able to remove personal traits
* or consider another architecture which can perform this function

#### Grid-search
* apply some basic filtering such as limits of loss ratios
* make some early stopping mechanisms and save models to check for convergence

#### Networks
* consider changing LSTM's to bidirectional
* consider adding convolutions in both generator and discriminator for locality
* make model more complex to learn arbitrary sequences more efficiently
* extend to RCGAN with realistic conditionings for actual usable data genration

#### Masking varied features
* come up with mask to create or ignore feature differences
* can be included within images

#### Input images:
* consider downsampling to save memory and computational power
* consider normalizing in a different way, via local max or possible integration
* plot input time series as normalized 2d images to show variation

#### Code-health:
* fix unused imports and sort with python tools
