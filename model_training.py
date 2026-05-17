import pydub
import pandas as pd
import numpy as np
from pathlib import Path
from pydub import AudioSegment as AS

import tensorflow as tf
from audio_dataset import AudioDataset
from classifier_model import ClassifierModel

import tqdm

def main():
    num_classes = 10
    batch_size = 8
    sr = 16000

    audio_dataset = AudioDataset(
        audio_dir = "classify_dataset",
        sample_rate = sr,
        num_classes = num_classes,
        batch_size = batch_size
    )

    X_batch, y_batch = audio_dataset[0]
    spectrogram_shape = X_batch.shape[1:]

    model = ClassifierModel(num_classes=num_classes)
    _ = model(tf.zeros((1, *spectrogram_shape)))
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    model.fit(audio_dataset, epochs=20)



if __name__ == "__main__":
    main()