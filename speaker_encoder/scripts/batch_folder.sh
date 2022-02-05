
#!/bin/bash


input_path="path_to_wavs_folder"
input_type="single_speaker"
num_wavs=-1 # -1 for computing embeddings of all wav files 
num_workers=8
output_name="output"
speaker_name="speaker_name"

python compute_embedding.py --input_path="$input_path"\
                            --input_type="$input_type"\
                            --num_wavs="$num_wavs"\
                            --num_workers="$num_workers"\
                            --output_name="$output_name" \
                            --speaker_name="$speaker_name"

