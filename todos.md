## Workflow changes

### Model stabilization and abstraction
* TODO: make pickles unglobbing more robust
* TODO: extract model and optimizer weights manually from colab run
* TODO: make pipeline adaptable to image pixels or modify dimensionality of lfw-faces -> test with lfw faces
* TODO: use stabilizing techniques such as noise addition to images, multi-scale gradient and spectral normalization
* consider using normally distributed label smoothing for richer distribution
* use dilations for larger scale temporal relations; might be necessary for images

### Model reconstruction and training continuation
* make function to temporally combine new trained model/archives with original run

### Cluster management
* make more efficient memory management script that can terminate processes and send emails
* sync results from cluster directly to computer
* try to share memory and tasks between gpus (multi\_gpu\_model)

### Backup Google Colab
* link google colab to local jupyter runtime
* run multiple notebooks directly from computer without browser
* sync google drive to localhost for easy access

### Heuristics
* add gradient checks for logging
* set up crowd-sourced grid-search via emails to check results
* optionally use tensorboard to analyze learning process
* look for similarity measure metrics which could be used alongside training

### Clean-code/documentation
* make intermediate documentation and add configuration for command line gpu usage to readme 
* track how many epochs or batch runs needed to converge and try to optimize this
* add conditions to `train.py` to add separate pipeline in RCGAN training

### Brainstorming points

#### GAN stabilisation:
* use stabilizing techniques such as: 
* label smoothing (done)
* differing learning rates for optimizers (done)
* noise addition to images
* spectral normalization
* multi-scale gradient

#### Evaluation pipeline
* use MIMIC data/models for direct TSTR/TRTS validations
* use TSTR/TRTS methodologies and identification issues to evaluate model
* combine various quality indicators to evaluate final model results
* explore privacy perspective and whether GAN is able to remove personal traits
* or consider another architecture which can perform this function


#### Networks and higher-dimensions abstraction
* extend to 64 pixels faces to check if abstraction possible
* make model more complex to learn arbitrary sequences more efficiently
* extend to RCGAN with realistic conditionings for actual usable data genration
* check out mathematical proofs for convergence on GAN's and relation to Nash equilibrium

#### Input images and feature masking
* come up with mask to create or ignore feature differences
* consider normalizing in a different way, via local max or possible integration
* plot input time series as normalized 2d images to show variation

#### Documentation and code-health:
* fix unused imports and sort with python tools
* encode proper documentation and model visualizations
