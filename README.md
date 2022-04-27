# Description

- Unofficial fastspeech2 TTS system, original paper: https://arxiv.org/pdf/2006.04558.pdf
- Created as part of a personal project as well as to practice implementing deep learning techniques.
- For inference and/or fine tuning on your own datasets see below
- Accompanying blog post on things I'v learnt implementing this paper is [here](https://torphix.github.io/blog/fastpages/jupyter/2022/01/24/Fastspeech.html)
# Setup

- Git clone this repository, cd into folder and create conda env with `conda env create -f conda_env.yaml`
- Initialise environment with `conda activate tts_fs2`

# Training
Note: all hyperparameters such as save locations, model features and data features can be modified by editing the files under fastspeech2/config
Defaults will suffice for testing, but if working with custom data worth having a look.

## Data preparation

### Option 1: Single Command Data processing pipeline
One helper method is provided that will download, prepare, align and create dataset all in one go: (Currently only [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset is supported)
 `python main.py fs2_make_dataset --dataset LJSpeech` 

### Option 2: Multi command data processing pipeline
Alternativly you can run each seperate command for a granular understanding of the data processing pipeline:
1. Download & extract dataset with command `python main.py download_dataset --dataset LJSpeech` (currently only LJSpeech supported)
2. Prepare dataset with command `python main.py fs2_prepare_dataset --dataset LJSpeech` or edit config file under fastspeech2/config/data.yaml
   - To use with your own dataset write a custom function under `fastspeech2/data/database/preprocessing/prepare_dataset.py`
   - Ensure that the folder structure is as such:
     - database
       - Custom
         - preprocessed_dataset
         - {speaker_id} (even if single speaker dataset)
           - .lab, .wav utterance pairs

3. Extract alignment from dataset with command `python main.py fs2_align_dataset` usually several error messages (due to lib used for aligning unfortunately nothing I can do about it) will appear but the alignment should work check database folder if unsure
4. Create dataset with command `python main.py fs2_create_dataset` this will preprocess the data by extracting audio features, phoneme durations and saving as pickled arrays for speedy loading during training.

## Training
- Train with command `python main.py fs2_train` after training models are saved to fastspeech2/saved_models and checkpoints are saved to checkpoints/fs2, provide the checkpoint path in config/trainer.yaml to resume training from checkpoint configure the model and trainer parameters in config directory
- To load training from checkpoint for finetuning or other use case either specify checkpoint path in config/trainer.yaml or use command
`python main.py fs2_train --ckpt_path={path/to/saved_checkpoint}`

# Inference

- 

# Lessons learnt from reading paper & building model
1. Finetuning different regions of the network to different tasks is a good way to increase time to convergance
  - FS2 has some modules learn pitch others energy and duration, I like to think of it like different regions of the brain specialising to various tasks
2. Need to mask padded tensors for convergance
  - My inital padding function was incorrect resulting in much wasted time and compute
3. Careful where you put activation functions, as can prevent convergance! or hit an asympotote much quicker
4. Adding residuals makes a big difference in minimum loss achievable, Using residuals in transformers causes loss asymptomte to reduce by half
5. Important to visualise and check all inputs and outputs of the model to ensure correct data is being fed to your model
6. Regularisation is extremely important, use residuals, dropout, gradient clipping and any other techniques necessary.
### Data
1. Build clean datapipelines
    Messy data -> Standardize to same format -> generate dataset -> input into model
    This means you only have to write code to serialize new 'messy' data into the standardized format
2. Do all the heavy lifting in the datapipeline, eg: feature calculation and formatting etc. this speeds up training time, dataset class should be solely for loading and returning data not
    processing

3. Pitch, energy & duration where not converging past a certain point:

4. Use tanh instead of relu at the end as negative values are required for inferencing the mel spec
5. Build up the complexity of the model graudally ie: see if you can fit to one speaker, then see if you can do multiple different speakers etc

### Biggest lessons to learn IMO:

- Extract additional features about the data as input to the model, this leads to faster convergance and may be necessary for convergance to begin with.
- Attempt to overfit on small subset of data when ever significant changes are made to see if that was what the error was due to
- Use an LR Scheduler! When hitting local optima in a high dimensional space it becomes very important to use LR Annealing 
### Notes
1. Due to my lack of compute I used a pretrained speaker encoder to extract the embedings for voices, the model used had been trained on 100s of voices over 4 large datasets
    git clone https://github.com/HamedHemati/SpeakerEncoder.git


### Open issue if you find a problem / desired added feature
This is a work in progress and updates will be added between my other projects
