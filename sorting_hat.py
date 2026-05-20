import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from audio_dataset import AudioDataset
from classifier_model import ClassifierModel

import os

import Preprocessing.clean as clean

instrument_classes = ["808", "clap", "flute", "hihat", "keys", "kick", "saxaphone", "snare", "violin", "vocals"]

# Sorts audio files into folders based on the predicted instrument class using the trained model
def sort(model, input_dir, output_dir):
    for instrument in instrument_classes:
        instrument_dir = output_dir / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in input_dir.rglob("*.wav"):
        # Same preprocessing steps as in clean.py
        target_rate = 16000
        threshold = 0.02
        window_seconds = 0.05
        clip_size = 0.25
        y, rate = clean.load_audio(path=audio_path, target_rate=target_rate)
        yf, mask, rolling_max = clean.filter(y, rate, threshold=threshold, window_size=window_seconds)
        if len(yf) == 0:
            continue
        clip = clean.split_audio(yf, target_rate, clip_size)

        # Same mel spectrogram conversion as in audio_dataset.py
        clip = tf.convert_to_tensor(clip, dtype=tf.float32)
        spectrogram = tf.signal.stft(clip, frame_length=256, frame_step=64)
        spectrogram = tf.abs(spectrogram)

        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = 64,
            num_spectrogram_bins = spectrogram.shape[-1],
            sample_rate = target_rate,
            lower_edge_hertz = 20.0,
            upper_edge_hertz = target_rate * 0.5
        )

        spectrogram = tf.matmul(tf.square(spectrogram), mel_weight_matrix)
        spectrogram = tf.math.log(spectrogram + 1e-5)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)

        # Add batch dimension
        spectrogram = tf.expand_dims(spectrogram, axis=0)

        # Predict instrument class
        predictions = model.predict(spectrogram, verbose=0)
        predicted_class_idx = np.argmax(predictions)
        predicted_instrument = instrument_classes[predicted_class_idx]

        # Move file to predicted instrument folder
        target_path = output_dir / predicted_instrument / audio_path.name
        os.rename(audio_path, target_path)
        print(f"Moved {audio_path.name} to {predicted_instrument} class folder")

def main():
    num_classes = 10
    sr = 16000 
    audio_dir = Path("classify_dataset")
    test_dir = Path("test_dataset")
    class_names = sorted([p.name for p in audio_dir.iterdir() if p.is_dir()])
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    test_paths = sorted(test_dir.rglob("*.wav"))
    test_dataset = AudioDataset(
        audio_paths = test_paths,
        class_to_idx = class_to_idx,
        sample_rate = sr,
        num_classes = num_classes,
        batch_size = 1,
        shuffle = False
    )

    model = ClassifierModel(num_classes=num_classes)
    X_batch, y_batch = test_dataset[0]
    spectrogram_shape = X_batch.shape[1:]
    _ = model(tf.zeros((1, *spectrogram_shape)))
    model.load_weights("model/best.weights.h5")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    sort(model, input_dir=Path("test_dataset_unsorted"), output_dir=Path("test_dataset_sorted"))

    print("Done sorting")
    
    return

if __name__ == "__main__":
    main()