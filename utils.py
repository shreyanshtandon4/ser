from logging import error
import tensorflow as tf
import librosa
import os
import numpy as np
from SpeechModel import SpeechModel
from CNNLTSMModel import CNNLSTMModel
EMOTION_DICT_RAVDESS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def process_audio_clip(file_path, label):
    file_path = file_path.numpy()
    audio, sr = librosa.load(file_path)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    chromagram = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0)
    extracted_features = tf.concat([mfcc, mel, chromagram, spectral, tonnetz], axis=0)
    return extracted_features, label


def get_dataset(
    training_dir="./training",
    label_dict=EMOTION_DICT_RAVDESS,
    validation_dir=None,
    val_split=0.2,
    batch_size=128,
    random_state=42,
    cache=True,
):

    def decompose_label(file_path: str):
        return label_to_int[file_path.split("-")[2]]

    def tf_wrapper_process_audio_clip(file_path, label):
        extracted_features, label = tf.py_function(
            process_audio_clip, [file_path, label], [tf.float32, tf.int32]
        )
        extracted_features.set_shape([193])
        label.set_shape([])
        extracted_features = tf.expand_dims(extracted_features, -1)
        return extracted_features, label

    file_path_list = os.listdir(training_dir)
    label_to_int = dict({(key, i) for i, key in enumerate(label_dict.keys())})
    labels = [decompose_label(file_path) for file_path in file_path_list]

    if validation_dir is None:
        if val_split > 0:
            from sklearn.model_selection import train_test_split

            file_path_list = [
                os.path.join(training_dir, path) for path in file_path_list
            ]
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                file_path_list, labels, test_size=val_split, random_state=random_state
            )
    else:
        train_paths = file_path_list
        train_labels = file_path_list
        val_paths = os.listdir(validation_dir)
        val_labels = [decompose_label(file_path) for file_path in val_paths]
        val_paths = [os.path.join(training_dir, path) for path in val_paths]

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    train_ds = train_ds.map(
        tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def create_model(num_output_classes):
    speechModel = SpeechModel(num_output_classes)
    model = speechModel.getRAVDESS()
    return model
