import tensorflow as tf

class ClassifierModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()

        # Three Conv2D layer pairs to extract increasinly complex features from the mel spectrogram
        # Two 2D pooling layers to reduce dimensionality
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding="same")
        self.pool_1 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same")
        self.pool_2 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same")

        # One Fully-Connected layer, then dropout to combat overfitting before outputting class predictions
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.2)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        # Connect everything sequentially (feature extraction -> learning)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.dropout_1(x, training=training)
        return self.out(x)