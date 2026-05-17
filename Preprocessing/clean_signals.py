import pydub
import pandas as pd
import numpy as np
from pathlib import Path
from pydub import AudioSegment as AS

import matplotlib.pyplot as plt
from scipy.io import wavfile
from glob import glob
import librosa
from librosa.core import resample, to_mono
import soundfile as sf

from tqdm import tqdm
import wavio
import os

"""
Data cleaning script: 
    - applies rolling average filter to remove long periods of silence
    - downsamples audio clips and converts to mono
    - makes each clip the same length by centering around the peak volume (padding with zeros if needed)
    - saves the mel spectrogram in a seperate folder (very cool)

"""

# Convert mp3 file to wav file for easier processing if needed 
def mp3_to_wav(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        print("No specified path")
        return 
    
    output_dir.mkdir(parents=True, exist_ok=True)
    mp3_files = list(input_dir.glob("*.mp3"))

    if not mp3_files:
        print("No .mp3 files found")
        return
    
    for mp3_file in mp3_files:
        wav_file = output_dir / f"{mp3_file.stem}.wav"
        print(f"Converting {mp3_file.name}")
        audio = AS.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")

# Apply a signal envelope to filter out the long empty parts of the sounds
def filter(y, rate, threshold, window_size):
    y_abs = pd.Series(y).apply(np.abs)
    ym = y_abs.rolling(window=int(rate * window_size), min_periods=1, center=True).max()
    mask = ym > threshold
    return y[mask.to_numpy()], mask.to_numpy(), ym.to_numpy()

# Make every sound clip the same length for consistency when converting to mel-spectrogram
# Finds the loudest portion of the clip and uses that as the center for window of target_size
def split_audio(y, rate, target_size):
    clip_len = int(rate * target_size)
    if len(y) == 0:
        return np.zeros(clip_len, dtype=np.float32)

    # Find the loudest point
    peak_idx = np.argmax(np.abs(y))
    start = peak_idx - clip_len // 2
    end = start + clip_len

    # Boundary cases
    if start < 0:
        start = 0
        end = clip_len
    if end > len(y):
        end = len(y)
        start = max(0, end - clip_len)
    
    # Pad the length of the clip with zeros if too short
    clip = y[start:end]
    if len(clip) < clip_len:
        padded = np.zeros(clip_len, dtype=np.float32)
        # Place shorter clip at the center of the entire clip
        pad_start = (clip_len - len(clip)) // 2
        padded[pad_start:pad_start + len(clip)] = clip
        clip = padded

    return clip.astype(np.float32)

# # Downsample audio and convert to mono
# def downsample_to_mono(target_rate, path):
#     wave = wavio.read(path).data.astype(np.float32, order='F')
#     try:
#         num_channels = wave.shape[1]
#         if num_channels == 2:
#             wave = to_mono(wave.T)
#         elif num_channels == 1:
#             wave = to_mono(wave.reshape(-1))
#     except IndexError:
#         wave = to_mono(wave.reshape(-1))
#         pass
#     wave = resample(wave, wave.rate, target_rate)
#     return wave.astype(np.int16)

# Saves the mel spectrogram of specified clip
def save_mel_spectrogram(audio_path, output):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Convert to mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024,hop_length=256,n_mels=128)

    # Convert power to db and export plot
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-Spectrogram for {audio_path}")
    plt.tight_layout()
    plt.savefig(os.path.join(output, f"{audio_path.stem}.png"))
    plt.close()

# Some of the wav files can have broken headers (for some annoying reason)
# So try using librosa, if that fails use pydub
def load_audio(path, target_rate):
    try:
        y, rate = librosa.load(path, sr=target_rate, mono=True)
        return y.astype(np.float32), rate

    except Exception as e:
        audio = AS.from_file(path)

        # Convert to mono and target sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_rate)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Normalize depending on sample width
        max_val = float(1 << (8 * audio.sample_width - 1))
        y = samples / max_val
        return y, target_rate

def main():
    input_path = "classify_dataset_unprocessed/vocals/"
    output_path = "classify_dataset/vocals/"

    # mp3_to_wav(input_dir=input_path, output_dir=output_path)

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 16kHz target sampling rate
    target_rate = 16000
    # Threshold for rolling window
    threshold = 0.02
    # Size of window
    window_seconds = 0.05
    # Size of each output clip
    clip_size = 0.25

    for audio_path in tqdm(list(in_path.glob("*.wav")), desc="Preprocessing audio"):
        y, rate = load_audio(path=audio_path, target_rate=target_rate)
        yf, mask, rolling_max = filter(y, rate, threshold=threshold, window_size=window_seconds)
        # If audio length is 0 after signal filter, skip
        if len(yf) == 0:
            continue
        
        # Split/pad audio centered around the loudest region in the clip
        clip = split_audio(yf, target_rate, clip_size)
        out = out_path / f"{audio_path.stem}.wav"
        sf.write(out, clip, target_rate)

        # Save output spectrogram
        save_mel_spectrogram(audio_path=out, output="mel spectrograms/vocals/")

    print("Done")
    return 

if __name__ == "__main__":
    main()