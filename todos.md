## Workflow to-do's

### Important
* make pipeline adaptable to image pixels or modify dimensionality of lfw-faces
* make function to continue training within train main file with new pickle
* continuing training function would need to load models, inter-relationships (trainables) and optimizer states
* make function to optionally combine new trained model with old one
* make function to continue training with arbitrary other instances, need to make workflow to sort old runs temporally and make necessary actions
* consider removing pickles directory necessity for greater flexibility and cli globbing
* check for possibility of saving and reconstructing models without weights pipeline

### Relevant concepts
* consider return sequences vs. return neurons
* make encoding and decoding lstm with cnn to compact architecture
* consider using stacked LSTMs
* track how many epochs or batch runs needed to converge and try to optimize this

### Heuristics
* set up crowd-sourced grid-search via emails to check results
* save optimizer to re-use when loading saved model
* add gradient checks and heuristics for early stopping mechanism
* add grid-search mechanism for checking more possibilities
* make mechanism for dynamic g-factor adjustment
* optionally use tensorboard to analyze learning process
* look for similarity measure metrics which could be used alongside training 

### Extra steps
* try to share memory and tasks between gpus (multi\_gpu\_model)
* take into account memory of system before running
* take into account gpu usage before executing
* add configuration for command line gpu usage to readme
* sync results from cluster directly to computer
* add workflow to log init files automatically without manual update of parameters
* default to single model class instead of dev/legacy when publishing
* consider allowing user to redefine parameters when continuing training and if this might be useful in application
* add pipeline to train/manage RCGAN in future steps

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
