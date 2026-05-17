import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from audio_dataset import AudioDataset
from classifier_model import ClassifierModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tqdm
import os

def main():
    num_classes = 10
    batch_size = 8
    sr = 16000 # 16 kHz (max frequency = 8 kHz)
    audio_dir = Path("classify_dataset")

    class_names = sorted([p.name for p in audio_dir.iterdir() if p.is_dir()])
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    audio_paths = sorted(audio_dir.rglob("*.wav"))
    labels = [class_to_idx[path.parent.name] for path in audio_paths]

    # Split data into train and val sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths,
        labels,
        test_size = 0.2, # 20% validation, 80% training
        random_state = 42,
        stratify = labels
    )

    train_dataset = AudioDataset(
        audio_paths = train_paths,
        class_to_idx = class_to_idx,
        sample_rate = sr,
        num_classes = num_classes,
        batch_size = batch_size,
        shuffle = True
    )

    val_dataset = AudioDataset(
        audio_paths = val_paths,
        class_to_idx = class_to_idx,
        sample_rate = sr,
        num_classes = num_classes,
        batch_size = batch_size,
        shuffle = False
    )

    model = ClassifierModel(num_classes=num_classes)

    # Match the dimension of the first Conv2D input layer from the model to the spectrogram shape
    X_batch, y_batch = train_dataset[0]
    spectrogram_shape = X_batch.shape[1:]
    _ = model(tf.zeros((1, *spectrogram_shape)))
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    # Save model with best validation accuracy
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="model/best.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1
    )

    # Early stopping when val loss begins to increase to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = "val_accuracy",
        patience = 8,
        mode = "max",
        restore_best_weights = True,
        verbose = 1
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join('history', '{}_history.csv'.format('model')),
        append = False
    )

    model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[model_checkpoint, early_stop, csv_logger])

if __name__ == "__main__":
    main()