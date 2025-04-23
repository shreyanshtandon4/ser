import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

class SpeechModel:
    def __init__(self, num_output_classes):
        self.num_output_classes = num_output_classes

    def getRAVDESS(self) -> Model:
        input_layer = L.Input(shape=(193, 1))

        # First convolutional layer
        x = L.Conv1D(filters=256, kernel_size=5, strides=1, padding='valid')(input_layer)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)

        # Second convolutional layer
        x = L.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(x)
        x = L.Activation('relu')(x)
        x = L.Dropout(0.1)(x)
        x = L.BatchNormalization()(x)

        # Max pooling layer
        x = L.MaxPooling1D(pool_size=8)(x)

        # Third convolutional layer
        x = L.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(x)
        x = L.Activation('relu')(x)

        # Fourth convolutional layer
        x = L.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(x)
        x = L.Activation('relu')(x)

        # Fifth convolutional layer
        x = L.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        x = L.Dropout(0.2)(x)

        # Flatten and dropout
        x = L.Flatten()(x)
        x = L.Dropout(0.2)(x)

        # Fully connected output layer
        x = L.Dense(self.num_output_classes)(x)
        x = L.BatchNormalization()(x)
        output = L.Activation('softmax')(x)

        # Define the model
        model = Model(inputs=input_layer, outputs=output)

        # RMSProp optimizer with specified learning rate and decay
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5, weight_decay=1e-6)

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
