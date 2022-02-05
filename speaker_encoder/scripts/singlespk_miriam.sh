
#!/bin/bash


input_path="/raid/hhemati/Datasets/Speech/TTS/VocallyYours/MiriamMeckel_Split/audios"
input_type="single_speaker"
num_wavs=-1
num_workers=8
output_name="miriam"
speaker_name="miriam"

python compute_embedding.py --input_path="$input_path"\
                            --input_type="$input_type"\
                            --num_wavs="$num_wavs"\
                            --num_workers="$num_workers"\
                            --output_name="$output_name" \
                            --speaker_name="$speaker_name"

