from sklearn.utils import validation
from tensorflow.python.keras.engine import training
from utils import get_dataset, create_model
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

train_dir = "./training"

val_dir = None

RANDOM_STATE = 42

val_split = 0.2

model_type = "ravdess"

NUM_LABELS = 8


train_ds, val_ds = get_dataset(
    training_dir=train_dir,
    validation_dir=val_dir,
    val_split=val_split,
    batch_size=32,
    random_state=RANDOM_STATE,
)

model = create_model(NUM_LABELS)

ESCallback = tf.keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True, verbose=3
)
model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=ESCallback,
    epochs=15,
)

model.save(f"saved_model/{EPOCHS}_trained_model.h5")



