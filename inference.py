# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import librosa
from tqdm import tqdm
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import demix_track, demix_track_demucs, get_model_from_config

import warnings
warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    print('Total files found: {}'.format(len(all_mixtures_path)))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)
    if args.store_dir == 'separation_results' and not os.path.isdir(args.store_dir + '/' + 'instruments'):
        os.mkdir(args.store_dir + '/' + 'instruments')
    if args.store_dir == 'karaoke_results' and not os.path.isdir(args.store_dir + '/' + 'karaoke'):
        os.mkdir(args.store_dir + '/' + 'karaoke')
    if args.store_dir == 'deverb_results' and not os.path.isdir(args.store_dir + '/' + 'reverb'):
        os.mkdir(args.store_dir + '/' + 'reverb')
    if args.store_dir == 'denoise_results' and not os.path.isdir(args.store_dir + '/' + 'noise'):
        os.mkdir(args.store_dir + '/' + 'noise')

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    for path in all_mixtures_path:
        if not verbose:
            all_mixtures_path.set_postfix({'track': os.path.basename(path)})
        try:
            # mix, sr = sf.read(path)
            mix, sr = librosa.load(path, sr=44100, mono=False)
            mix = mix.T
        except Exception as e:
            print('Can read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if args.model_type == 'htdemucs':
            res = demix_track_demucs(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)
        for instr in instruments:
            sf.write("{}/{}_{}.wav".format(args.store_dir, os.path.basename(path)[:-4], instr), res[instr].T, sr, subtype='FLOAT')

        if 'Vocals' in instruments and args.extract_instrumental:
            instrum_file_name = "{}/{}/{}_{}.wav".format(args.store_dir, 'instruments', os.path.basename(path)[:-4], 'Instrumental')
            sf.write(instrum_file_name, mix - res['Vocals'].T, sr, subtype='FLOAT')
        elif 'vocals' in instruments and args.extract_instrumental:
            instrum_file_name = "{}/{}/{}_{}.wav".format(args.store_dir, 'instruments', os.path.basename(path)[:-4], 'Instrumental')
            sf.write(instrum_file_name, mix - res['vocals'].T, sr, subtype='FLOAT')

        if 'karaoke' in instruments and args.extract_karaoke:
            instrum_file_name = "{}/{}/{}_{}.wav".format(args.store_dir, 'karaoke', os.path.basename(path)[:-4], 'other')
            sf.write(instrum_file_name, mix - res['karaoke'].T, sr, subtype='FLOAT')
            
        if 'noreverb' in instruments and args.extract_reverb:
            instrum_file_name = "{}/{}/{}_{}.wav".format(args.store_dir, 'reverb', os.path.basename(path)[:-4], 'reverb')
            sf.write(instrum_file_name, mix - res['noreverb'].T, sr, subtype='FLOAT')
            
        if 'dry' in instruments and args.extract_noise:
            instrum_file_name = "{}/{}/{}_{}.wav".format(args.store_dir, 'noise', os.path.basename(path)[:-4], 'noise')
            sf.write(instrum_file_name, mix - res['dry'].T, sr, subtype='FLOAT')

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true', help="invert vocals to get instrumental if provided")
    parser.add_argument("--extract_karaoke", action='store_true', help="invert lead vocal to get karaoke if provided")
    parser.add_argument("--extract_reverb", action='store_true', help="invert noreverb to get reverb if provided")
    parser.add_argument("--extract_noise", action='store_true', help="invert denoise to get noise if provided")
    parser.add_argument("--force_cpu", action = 'store_true', help = "Force the use of CPU even if CUDA is available")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
        
    use_cuda = torch.cuda.is_available() and not args.force_cpu

    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        if use_cuda:
            state_dict = torch.load(args.start_check_point)
        else:
            state_dict = torch.load(args.start_check_point, map_location = torch.device('cpu'))
        if args.model_type == 'htdemucs':
            # Fix for htdemucs pround etrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    if use_cuda:
        device_ids = args.device_ids
        if type(device_ids) == int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids = device_ids).to(device)
        print('Using CUDA with device_ids: {}'.format(device_ids))
    else:
        device = 'cpu'
        print('Using CPU. It will be very slow!')
        print('If CUDA is available, use --force_cpu to disable it.')
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
