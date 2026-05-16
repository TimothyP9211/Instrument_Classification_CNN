import os
from pathlib import Path

import tensorflow as tf
import numpy as np

class AudioDataset(tf.keras.utils.Sequence):
    # Dataloader for wav files
    def __init__(self, audio_dir, sample_rate, num_classes, batch_size, shuffle=True):
        self.audio_dir = Path(audio_dir)

        # Audio paths
        self.audio_paths = sorted([
            p for p in self.audio_dir.rglob("*.wav")
        ])

        # Class names (same as parent folder names)
        self.class_names = sorted([
            p.name for p in self.audio_dir.iterdir()
            if p.is_dir()
        ])

        self.class_to_idx = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }

        self.sr = sample_rate
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.audio_paths) / self.batch_size))

    # Get mel spectrograms and labels of batch index=idx
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        X, y = [],[]
        for audio_path in self.audio_paths[start:end]:
            # Read and decode wav file binary
            audio_binary = tf.io.read_file(str(audio_path))
            audio, sr = tf.audio.decode_wav(audio_binary)

            # Convert from (samples, channels=1) to (samples,)
            audio = tf.squeeze(audio, axis=-1)

            # Create the mel spectrogram and add a dimension for channel
            spectrogram = tf.abs(tf.signal.stft(audio, frame_length=1024, frame_step=256))
            spectrogram = tf.expand_dims(spectrogram, axis=-1)
            X.append(spectrogram)

            # Get the labels from parent folder names
            class_name = audio_path.parent.name
            label_idx = self.class_to_idx[class_name]
            label = tf.keras.utils.to_categorical(label_idx, num_classes=self.num_classes)
            y.append(label)

        X = tf.stack(X)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.audio_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
