from .limit_threads import *
import argparse
import glob
import os
import numpy as np
import pickle
import random
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from .speaker_encoder.model import SpeakerEncoder
from .speaker_encoder.audio import AudioProcessor
from .speaker_encoder.io import load_config


class SpeechEmbedding():
    def __init__(self, config, model_path):
        self.ap = AudioProcessor(**config['audio'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Encoder model and load pretrained checkpoint
        self.model = SpeakerEncoder(**config.model).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model'])
        self.model.eval()

    def compute_embedding(self, wav_file, itr, total, verbose=True):
        if verbose:
            print(f"Computing embedding for file {itr}/{total}")

        mel_spec = self.ap.melspectrogram(self.ap.load_wav(wav_file, sr=self.ap.sample_rate)).T
        mel_spec = torch.FloatTensor(mel_spec[None, :, :])
        mel_spec = mel_spec.to(self.device)
        embedd = self.model.compute_embedding(mel_spec)
        embedd = embedd.detach().cpu().numpy()

        return embedd
    

def main(args):
    config = load_config(args.config_path)
    speech_embedding = SpeechEmbedding(config, args.model_path)

    emb_dict = {}
    # Compute speaker embeddings
    if args.input_type == "single_file":
        wav_file = args.input_path
        embedd = speech_embedding.compute_embedding(wav_file, 1, 1)
        embedd = embedd[0]
        emb_dict[args.speaker_name] = embedd
        print(embedd)
    else:
        if args.input_type == "single_speaker":
            spk_list = [args.speaker_name]
        elif args.input_type == "multi_speaker":
            spk_list = os.listdir(args.input_path)

        executor = ProcessPoolExecutor(max_workers=args.num_workers)
        for spk_itr, spk_name in enumerate(spk_list):
            print(f"========== Speaker {spk_itr}/{len(spk_list)}::")
            if args.input_type == "single_speaker":
                wav_files = glob.glob(os.path.join(args.input_path, "*.wav"))
            elif args.input_type == "multi_speaker":
                wav_files = glob.glob(os.path.join(args.input_path, spk_name, "*.wav"))
            
            # Skip if no wav available for speaker
            if len(wav_files) == 0:
                continue

            # Randomly shuffle and select a sub-list of num_wavs != -1
            if args.num_wavs != -1:
                print(f"Selecting {args.num_wavs} random wavs ...")
                random.shuffle(wav_files)
                wav_files = wav_files[:args.num_wavs]
            
            # Compute embeddings for all wav files
            all_embdds = []
            for itr, wav_file in enumerate(wav_files):
                embedd = executor.submit(speech_embedding.compute_embedding, 
                                         wav_file,
                                         itr,
                                         len(wav_files))
                all_embdds.append((os.path.basename(wav_file), embedd))

            # Process outputs
            all_embdds = [(embedd[0], embedd[1].result()) for embedd in all_embdds if embedd[1].result() is not None]
            
            # Add embedding of all files
            emb_dict[spk_name] = {}
            if args.mode == "all_embs":
                emb_dict[spk_name].update({embed[0]:embed[1][0] for embed in all_embdds})

            # Add mean of embeddings
            all_embdds_list = [list(embedd[1][0]) for embedd in all_embdds]
            embedd_mean = np.mean(np.array(all_embdds_list), axis=0)
            emb_dict[spk_name].update({"mean":embedd_mean})

    with open(os.path.join(args.output_path, f"{args.output_name}_emb.pkl"), "wb") as pkl_file:
        pickle.dump(emb_dict, pkl_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="pretrained_model/best_model.pth.tar", required=False)
    parser.add_argument('--config_path', type=str, default="pretrained_model/config.json", required=False)
    parser.add_argument('--output_path', type=str, default="outputs/", required=False)
    parser.add_argument('--input_path', type=str, default="outputs/", required=False)
    parser.add_argument('--input_type', type=str, default="file", required=False)  # single_speaker, #multi_speaker
    parser.add_argument('--speaker_name', type=str, default="default", required=False)  
    parser.add_argument('--num_wavs', type=int, default=20, required=False)
    parser.add_argument('--num_workers', type=int, default=10, required=False)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--mode', type=str, default="all_embs", required=False)

    args = parser.parse_args()

    main(args)