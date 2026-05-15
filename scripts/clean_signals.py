import pydub
import pandas as pd
import numpy as np
from pathlib import Path
from pydub import AudioSegment as AS

import matplotlib.pyplot as plt
from scipy.io import wavfile
from glob import glob
from librosa.core import resample, to_mono

from tqdm import tqdm
import wavio
import os

in_path = "classify_dataset_unprocessed/808"
out_path = "classify_dataset/808"

# Convert mp3 file to wav file for easier processing
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
def filter(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    ym = y.rolling(window=int(rate=20), min_periods=1, center=True).max()
    for m in ym:
        m.append(m > threshold)
    return m, ym

def main():
    mp3_to_wav(input_dir=in_path, output_dir=out_path)

    # TODO: Clean out each .mp3 file using a rolling signal envelope and export as wav files


if __name__ == "__main__":
    main()