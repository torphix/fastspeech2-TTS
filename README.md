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
 `python main.py make_dataset --dataset LJSpeech`

### Option 2: Multi command data processing pipeline (Custom datasets)
Alternativly you can run each seperate command if you are finetuning on custom dataset:
1. Download & extract dataset with command `python main.py download_dataset --dataset LJSpeech` (currently only LJSpeech supported)
2. Prepare dataset with command `python main.py prepare_dataset --dataset LJSpeech` or edit config file under fastspeech2/config/data.yaml
   - To use with your own dataset write a custom function under `fastspeech2/data/database/preprocessing/prepare_dataset.py`
   - Ensure that the folder structure is as such:
     - database
       - DATASET_NAME (must match name in fastspeech2/configs/data.yaml)
         - preprocessed_dataset
         - {speaker_id} (even if single speaker dataset)
           - .lab, .wav text, audio pairs

3. Extract alignment from dataset with command `python main.py align_dataset` usually several error messages (due to lib used for aligning unfortunately nothing I can do about it) will appear but the alignment should work check database folder if unsure (will be populated with TextGrid folder + files)
4. Create dataset with command `python main.py create_dataset` this will preprocess the data by extracting audio features, phoneme durations and saving as pickled arrays for speedy loading during training.

## Training
- Train with command `python main.py train` after training models are saved to fastspeech2/saved_models and checkpoints are saved to checkpoints/fs2, provide the checkpoint path in config/trainer.yaml to resume training from checkpoint configure the model and trainer parameters in config directory
- To load training from checkpoint for finetuning or other use case either specify checkpoint path in config/trainer.yaml or use command
`python main.py train --ckpt_path={path/to/saved_checkpoint}`

# Inference

- `python main.py inference --text='I need your clothes, your boots, and your motorcycle' --model_path='path/to/model.pth.tar` pretrained model is [here]()
Leave model_path blank to inference with default 
Trained models are saved to fastspeech2/trained_models


# Lessons learnt from reading paper & building model
1. Localise training of various different sections to downstream tasks here pitch and energy was first extracted and submodules trained to predict those features based on the phoneme input before predicting the mel spectorgram
2. Need to mask padded tensors for convergance
  - My inital padding function was incorrect resulting in much wasted time and compute
3. Regularisation is super important adding residuals makes a big difference in minimum loss achievable, Using residuals in transformers causes loss asymptomte to reduce by half
4. Learning rate annealing is also important for efficient and reasonable convergance
5. Attempt to overfit on small subset of data when ever significant changes are made to see if that was what the error was due to


### Open issue if you find a problem / desired added feature
This is a work in progress and updates will be added between my other projects
