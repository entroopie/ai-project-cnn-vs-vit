from time import time

from tensorflow import keras
from tensorflow.keras import layers

# --- Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS = 2
DROPOUT_CONV = 0.3
DROPOUT_FC = 0.4
DENSE_UNITS = 128
MODEL_PATH = "models/rice_cnn.keras"
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-6


def build_cnn_model(img_size, num_classes):
    """
    Input:  img_size: tuple[int, int], num_classes: int
    Output: model: keras.Sequential
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(*img_size, 3)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(DROPOUT_CONV),
            layers.Flatten(),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dropout(DROPOUT_FC),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(model, train_ds, val_ds, class_weight):
    """
    Input:  model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, class_weight: dict[int, float]
    Output: (history: keras.callbacks.History, train_time: float)
    """
    callbacks = [
        keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR),
    ]

    start_time = time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    train_time = time() - start_time
    return history, train_time
