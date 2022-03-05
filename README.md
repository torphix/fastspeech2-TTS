# Description

- Unofficial fastspeech2 TTS system, original paper: https://arxiv.org/pdf/2006.04558.pdf
- Created as part of a personal project as well as to practice implementing deep learning techniques.
- For inference and/or fine tuning on your own datasets see below
- Accompanying blog post on things I'v learnt implementing this paper is [here] (https://www.genome.gov/)

# Setup

- Git clone this repository, cd into folder and create conda env with `conda env create -f conda_env.yml`

# Training

You can use the notebook provided as TTS.ipynb in the root folder or alternativly use the command line.

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

6. Finetuning different regions of the network to different tasks is a good way to increase time to convergance
7. Need to mask padded tensors for convergance
8. If model is not converging, see if loss reduces when replacing a particular module eg: transformer with plain convs,
   if so this indicates that the module is broken
9. Careful where you put activation functions, as can prevent convergance! or hit an asympotote much quicker
10. Adding residuals makes a big difference in minimum loss achievable, Using res in transformer causes loss asymptomte to reduce by half
11. LR effects different parts of then network, mel will converge at LR of 1e-3 whilst duration wont converge unless its at 1e-4
    pitch wont converge unless at 1e-4 and energy will converge at 1e-3,
12. Dropout is also important, all regularization techniques is important
<!-- Data -->
13. Build clean datapipelines
    Messy data -> Standardize to same format -> generate dataset -> input into model
    This means you only have to write code to serialize new 'messy' data into the standardized format
14. Do all the heavy lifting in the datapipeline, eg: feature calculation and formatting etc. this speeds up training time, dataset class should be solely for loading and returning data not
    processing

15. Pitch, energy & duration where not converging past a certain point:

16. Use tanh instead of relu at the end as negative values are required for inferencing the mel spec
17. Build up the complexity of the model graudally ie: see if you can fit to one speaker, then see if you can do multiple different speakers etc

Biggest lessons to learn IMO:

- Extract additional features about the data as input to the model, this leads to faster convergance and may be necessary for convergance to begin with.
- Attempt to overfit on small subset of data when ever significant changes are made to see if that was what the error was due to

11. Due to my lack of compute I used a pretrained speaker encoder to extract the embedings for voices, the model used had been trained on 100s of voices over 4 large datasets
    git clone https://github.com/HamedHemati/SpeakerEncoder.git

12. Ensure data is correct and normalised between [-1,1]

TODO:

1. Drop too long sentances, order similar length batchs together for training to prevent wasted computations
2. Change to phoneme level prediction as == better prosody
3. Setup inference, test on unseen data and test inference
4. Try more data
5. Split configs into seperate files

Data:

6. TODO: - Standard way to format data for inputting and processing
7. Batchify speaker encoding operation

   How To:
   Train on LibriSpeech corpus

8. Download LibriSpeech Corpus and unzip into data/raw_data/LibriSpeech, add the path to libri_speech.yaml config
9. Call python3 main.py prepare_dataset --config ./config/libri_speech.yaml
10. Install Montreal Forced Alignment to extract the durations of each phoneme
11. Align by calling python3 main.py align_dataset --type librispeech

<!-- Go From Here -->

5. Generate dataset by calling python3 main.py create_dataset --config ./config/libri_speech.yaml --feature_level 'phoneme_frame' or 'mel_frame'
6. Train by calling python3 main.py train --config ./config/libri_speech.yaml --trainer_config ./config/trainer.yaml

NEXT:

1.  Vocoder, test with first then fix up fs2 code base and write blog so you can apply to masters
2.  Then continue with vocoder

TODO:
Only need to compute 1 speaker embedding not duplicate speaker embeddings for different speakers

TRY: (If not working)

1. min max scaling instead of mean std scaling as may converge better if values between 0 and 1
2. With standard scaling make sure you compute metadata across entire dataset FIRST as this ensures your using the true mean and std not just the speaker mean and std

TODO:

- Vecotrise portaspeech pos encoding
