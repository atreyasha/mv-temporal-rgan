## Workflow to-do's

### Important
* run model on mnist, fashion_mnist and then faces with differing epochs
* pickle detailed log files and models; create within main file such that class reconstruction is possible
* redefine model and integrate dropout into GPU implementation
* add dropout to gpu model configuration
* add batch normalization for better stability

### Conceptual issues
* think more about lstm returning seq or neurons for generation process
* perhaps keep same timesteps and return sequences and later use cnn to reduce dimensions with global pooling
* make encoding and decoding lstm with cnn to compact architecture

### Next steps
* add gradient checks and heuristics for early stopping mechanism
* add grid-search mechanism for checking more possibilities
* make mechanism for dynamic g-factor adjustment
* configure code to use specific gpus on cluster
* try to share memory and tasks between gpus (multi\_gpu\_model)
* take into account memory of system before running
* take into account gpu usage before executing
* optionally use tensorboard to analyze learning process
* find out how to make correlated time series with LSTM

### Brainstorming points

#### Evaluation pipeline
* train mnist, fashion-mnist and lfw-faces for 28 pixels
* extend to 64 pixels faces to check if abstraction possible
* use TSTR/TRTS methodologies and identification issues to evaluate model
* use MIMIC data/models for direct TSTR/TRTS validations

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
