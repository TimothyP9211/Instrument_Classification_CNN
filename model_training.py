import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from audio_dataset import AudioDataset
from classifier_model import ClassifierModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import os

# Saves the normalized confusion matrix for the test set predictions
def save_confusion_matrix(model, test_dataset, class_names, output_path="confusion_matrix.png"):
    y_true = []
    y_pred = []

    for X_batch, y_batch in test_dataset:
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        values_format=".2f"
    )

    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved normalized confusion matrix to {output_path}")

def main():
    num_classes = 10
    batch_size = 8
    sr = 16000 # 16 kHz (max frequency = 8 kHz)
    audio_dir = Path("classify_dataset")
    test_dir = Path("test_dataset")

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

    test_paths = sorted(test_dir.rglob("*.wav"))

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

    test_dataset = AudioDataset(
        audio_paths = test_paths,
        class_to_idx = class_to_idx,
        sample_rate = sr,
        num_classes = num_classes,
        batch_size = 1,
        shuffle = False
    )

    # MODEL TRAINING

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

    # MODEL EVALUATION

    # Load the model with best validation accuracy 
    best_model = ClassifierModel(num_classes=num_classes)
    _ = best_model(tf.zeros((1, *spectrogram_shape)))
    best_model.load_weights("model/best.weights.h5")

    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Evaluate model on test set 
    test_loss, test_acc = best_model.evaluate(test_dataset)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    save_confusion_matrix(
        best_model,
        test_dataset,
        class_names,
        output_path="confusion_matrix.png"
    )

if __name__ == "__main__":
    main()