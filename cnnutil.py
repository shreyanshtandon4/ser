import tensorflow as tf
import librosa
import os
import numpy as np

# Emotion dictionary for RAVDESS
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
    file_path = file_path.numpy().decode("utf-8")
    try:
        audio, sr = librosa.load(file_path, sr=None)
        max_len = 3 * sr  # Pad or trim to 3 seconds
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]

        # Extract MFCC (or any consistent representation)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T  # (time_steps, features)

        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

        return mfcc.astype(np.float32), label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((130, 40), dtype=np.float32), label


def get_dataset(
    training_dir="./training",
    label_dict=EMOTION_DICT_RAVDESS,
    validation_dir=None,
    val_split=0.2,
    batch_size=64,
    random_state=42,
    cache=True,
):
    label_to_int = {k: i for i, k in enumerate(label_dict.keys())}

    def decompose_label(file_path: str):
        return label_to_int[file_path.split("-")[2]]

    def tf_wrapper_process_audio_clip(file_path, label):
        features, label = tf.py_function(
            process_audio_clip, [file_path, label], [tf.float32, tf.int32]
        )
        features.set_shape([None, 40])
        label.set_shape([])
        return features, label

    file_names = os.listdir(training_dir)
    full_paths = [os.path.join(training_dir, f) for f in file_names]
    labels = [decompose_label(f) for f in file_names]

    if validation_dir is None:
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            full_paths, labels, test_size=val_split, random_state=random_state
        )
    else:
        val_names = os.listdir(validation_dir)
        val_paths = [os.path.join(validation_dir, f) for f in val_names]
        val_labels = [decompose_label(f) for f in val_names]
        train_paths, train_labels = full_paths, labels

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    train_ds = train_ds.map(tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
def create_model(num_output_classes):

    from CNNLTSMModel import CNNLSTMModel

    # Choose the model you want to use
    # model = SpeechModel(num_output_classes).getRAVDESS()
    model = CNNLSTMModel(num_output_classes).get_model()

    return model