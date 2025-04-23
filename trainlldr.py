import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split

# 1. Configuration
DATA_DIR = "./training"  # path to RAVDESS audio files (wav)
SR = 16000  # sample rate for audio
DURATION = 3  # seconds (RAVDESS clips ~3 sec)
NUM_CLASSES = 8  # emotions count

BATCH_SIZE = 16
EPOCHS = 50
BASE_LR = 1e-3
DECAY = 0.8
RANDOM_STATE = 42

# 2. Emotion mapping from filename
# RAVDESS filenames are like: 01-01-01-01-01-01-01.wav
# 3rd number = emotion:
# 1=neutral,2=calm,3=happy,4=sad,5=angry,6=fearful,7=disgust,8=surprised
EMOTION_MAP = {
    '01': 0,  # neutral
    '02': 1,  # calm
    '03': 2,  # happy
    '04': 3,  # sad
    '05': 4,  # angry
    '06': 5,  # fearful
    '07': 6,  # disgust
    '08': 7,  # surprised
}

# 3. Load filenames and labels
def load_ravdess_filepaths_labels(data_dir):
    filepaths = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                # parse emotion from filename
                parts = file.split("-")
                emotion_code = parts[2]
                label = EMOTION_MAP.get(emotion_code)
                if label is not None:
                    filepaths.append(os.path.join(root, file))
                    labels.append(label)
    return filepaths, labels

filepaths, labels = load_ravdess_filepaths_labels(DATA_DIR)
print(f"Loaded {len(filepaths)} files")

# 4. Audio preprocessing: load wav, pad/cut, extract log-mel spectrogram
def preprocess_audio(file_path, sr=SR, duration=DURATION, n_mels=64, n_fft=1024, hop_length=512):
    # Load audio
    wav, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
    # Pad or cut to fixed length
    expected_len = sr * duration
    if len(wav) < expected_len:
        wav = np.pad(wav, (0, expected_len - len(wav)))
    else:
        wav = wav[:expected_len]

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Log scaling
    log_mel_spec = librosa.power_to_db(mel_spec)
    # Normalize to 0 mean, 1 std
    norm_log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-9)
    # Transpose to time x freq (so time dimension is sequence length for LSTM)
    return norm_log_mel_spec.T.astype(np.float32)  # shape: (time_steps, n_mels)

# 5. Create dataset arrays
X = np.array([preprocess_audio(fp) for fp in filepaths])
y = np.array(labels)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# 6. Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 7. Create tf.data.Dataset
def create_dataset(X, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(X_train, y_train)
val_ds = create_dataset(X_val, y_val, shuffle=False)

# 8. CNN + LSTM model definition
def create_cnn_lstm_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)  # (time_steps, n_mels)

    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

input_shape = X_train.shape[1:]  # (time_steps, n_mels)
model = create_cnn_lstm_model(input_shape, NUM_CLASSES)
model.summary()

# 9. LLDR optimizer wrapper (Adam)
class LLDRAdam(tf.keras.optimizers.Adam):
    def __init__(self, lr_multipliers, base_lr=1e-3, **kwargs):
        super().__init__(learning_rate=base_lr, **kwargs)
        self.lr_multipliers = lr_multipliers

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        scaled_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is None:
                scaled_grads_and_vars.append((grad, var))
                continue
            multiplier = self.lr_multipliers.get(var.name, 1.0)
            scaled_grad = grad * multiplier
            scaled_grads_and_vars.append((scaled_grad, var))
        super().apply_gradients(scaled_grads_and_vars, name, experimental_aggregate_gradients)

# 10. Generate LR multipliers (decayed per layer)
def get_lr_multipliers(model, decay=DECAY):
    n_layers = len(model.layers)
    multipliers = {}
    for i, layer in enumerate(model.layers):
        mult = decay ** (n_layers - 1 - i)
        for var in layer.trainable_variables:
            multipliers[var.name] = mult
    return multipliers

lr_multipliers = get_lr_multipliers(model, decay=DECAY)

# 11. Compile model
optimizer = LLDRAdam(lr_multipliers=lr_multipliers, base_lr=BASE_LR)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),'accuracy']
)

# 12. Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, verbose=1
)

# 13. Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
    
)

# 14. Save model
model.save("saved_model/cnn_lstm_ravdess_llrd.h5")
