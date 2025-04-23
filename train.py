from cnnutil import get_dataset, create_model
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
train_dir = "./training"
RANDOM_STATE = 42
val_split = 0.2
NUM_LABELS = 8
train_ds, val_ds = get_dataset(
    training_dir=train_dir,
    val_split=val_split,
    batch_size=16,
    random_state=RANDOM_STATE,
    cache=True,
)
model = create_model(NUM_LABELS)

ESCallback = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, verbose=3
)
model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[ESCallback],
    epochs=80,
)
model.save("saved_model/trained_model_dm.h5")
