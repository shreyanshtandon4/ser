import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

class CNNLSTMModel:
    def __init__(self, num_output_classes):
        self.num_output_classes = num_output_classes

    def get_model(self) -> Model:
        input_layer = L.Input(shape=(193, 40))  # (timesteps, features)

        # CNN feature extractor
        x = L.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(input_layer)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        x = L.MaxPooling1D(pool_size=2)(x)

        x = L.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        x = L.MaxPooling1D(pool_size=2)(x)

        x = L.Dropout(0.3)(x)

        # LSTM for temporal modeling
        x = L.LSTM(64, return_sequences=True)(x)
        x = L.LSTM(32)(x)

        # Dense classifier
        x = L.BatchNormalization()(x)
        x = L.Dense(64, activation='relu')(x)
        x = L.Dropout(0.3)(x)

        output = L.Dense(self.num_output_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output)

        # Optimizer
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
