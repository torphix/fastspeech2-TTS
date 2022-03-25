# Description

- Unofficial fastspeech2 TTS system, original paper: https://arxiv.org/pdf/2006.04558.pdf
- Created as part of a personal project as well as to practice implementing deep learning techniques.
- For inference and/or fine tuning on your own datasets see below
- Accompanying blog post on things I'v learnt implementing this paper is [here](https://torphix.github.io/blog/fastpages/jupyter/2022/01/24/Fastspeech.html)
# Setup

- Git clone this repository, cd into folder and create conda env with `conda env create -f conda_env.yml`

# Training

- Command line:

One helper method is provided that will download, prepare, align and create dataset all in one go: `python main.py fs2_make_dataset --dataset {$NAME_OF_DATASET}` alternatively you can run the individual methods.

1. Download & extract dataset with command `python main.py download_dataset --dataset LJSpeech` (currently only LJSpeech supported)
2. Prepare dataset with command `python main.py fs2_prepare_dataset --dataset LJSpeech` or edit config file under fastspeech2/config/data.yaml

   - To use with your own dataset write a custom function under `fastspeech2/data/database/preprocessing/prepare_dataset.py`
   - Ensure that the folder structure is as such:
     - database
       - Custom
         - preprocessed_dataset
         - {speaker_id} (even if single speaker dataset)
           - .lab, .wav utterance pairs

3. Extract alignment from dataset with command `python main.py fs2_align_dataset` usually several error messages will appear but the alignment should work check database folder if unsure
4. Create dataset with command `python main.py fs2_create_dataset`
5. Train with command `python main.py fs2_train` after training models are saved to fastspeech2/saved_models and checkpoints are saved to checkpoints/fs2, provide the checkpoint path in config/trainer.yaml to resume training from checkpoint
   Configure the model and trainer parameters in config directory

# Inference

Checkpoint can be found: [here](https://drive.google.com/file/d/1dcIFZCn1aRu46dX1lfa3CDzoM0D-3VJ0/view?usp=sharing)
Audio generated is understandable but not excellent, further training and expansion of the model capacity should equate to better results.

1. Finetuning different regions of the network to different tasks is a good way to increase time to convergance
2. Need to mask padded tensors for convergance
3. If model is not converging, see if loss reduces when replacing a particular module eg: transformer with plain convs,
   if so this indicates that the module is broken
4. Careful where you put activation functions, as can prevent convergance! or hit an asympotote much quicker
5. Adding residuals makes a big difference in minimum loss achievable, Using res in transformer causes loss asymptomte to reduce by half
6. LR effects different parts of then network, mel will converge at LR of 1e-3 whilst duration wont converge unless its at 1e-4
    pitch wont converge unless at 1e-4 and energy will converge at 1e-3,
7. Dropout is also important, all regularization techniques is important
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
- 
### Notes
1. Due to my lack of compute I used a pretrained speaker encoder to extract the embedings for voices, the model used had been trained on 100s of voices over 4 large datasets
    git clone https://github.com/HamedHemati/SpeakerEncoder.git


### Open issue if you find a problem / desired added feature
This is a work in progress and updates will be added between my other projects
