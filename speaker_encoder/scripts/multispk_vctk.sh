#!/bin/bash


input_path="/raid/hhemati/Datasets/Speech/TTS/English/VCTK-Corpus/wavs/"
input_type="multi_speaker"
num_wavs=-1
num_workers=8
output_name="vctk"


python compute_embedding.py --input_path="$input_path"\
                            --input_type="$input_type"\
                            --num_wavs="$num_wavs"\
                            --num_workers="$num_workers"\
                            --output_name="$output_name"

