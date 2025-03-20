import tensorflow as tf
import keras
from keras import layers as L
from keras import Model

class SpeechModel:
    def __init__(self, num_output_classes) -> None:
        self.num_output_classes = num_output_classes

    def getRAVDESS(self) -> Model:
        input_layer = L.Input(shape=(193, 1))

        cnn1 = L.Conv1D(256, (5))(input_layer)
        batch_norm1 = L.BatchNormalization()(cnn1)
        relu1 = L.ReLU()(batch_norm1)

        cnn2 = L.Conv1D(128, (5))(relu1)
        relu2 = L.ReLU()(cnn2)
        dropout1 = L.Dropout(0.1)(relu2)
        batch_norm2 = L.BatchNormalization()(dropout1)

        max_pool1 = L.MaxPool1D(8)(batch_norm2)

        conv3 = L.Conv1D(128, (5))(max_pool1)
        relu3 = L.ReLU()(conv3)
        conv4 = L.Conv1D(128, (5))(relu3)
        relu4 = L.ReLU()(conv4)
        conv5 = L.Conv1D(128, (5))(relu4)
        batch_norm4 = L.BatchNormalization()(conv5)
        relu5 = L.ReLU()(batch_norm4)
        dropout2 = L.Dropout(0.2)(relu5)

        conv6 = L.Conv1D(128, (5))(dropout2)
        flatten = L.Flatten()(conv6)
        dropout3 = L.Dropout(0.2)(flatten)

        output_logits = L.Dense(self.num_output_classes)(dropout3)
        batch_norm5 = L.BatchNormalization()(output_logits)
        softmax = L.Softmax()(batch_norm5)
        model = Model(inputs=[input_layer], outputs=[softmax])
        optimizer = tf.keras.optimizers.RMSprop(1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss)

        return model