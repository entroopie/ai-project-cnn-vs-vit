from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS = 20
PATCH_SIZE = 16
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 6
MLP_DIM = 128
DROPOUT_RATE = 0.1
MODEL_PATH = "models/rice_vit.keras"
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-6


def mlp(x, hidden_units):
    """
    Input:  x: tf.Tensor shape (B, N, D), hidden_units: list[int]
    Output: x: tf.Tensor shape (B, N, hidden_units[-1])
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
    return x


def build_vit_model(img_size, num_classes):
    """
    Input:  img_size: tuple[int, int], num_classes: int
    Output: model: keras.Model
    """
    num_patches = (img_size[0] // PATCH_SIZE) * (img_size[1] // PATCH_SIZE)

    inputs = layers.Input(shape=(*img_size, 3))
    patches = layers.Conv2D(
        PROJECTION_DIM,
        kernel_size=PATCH_SIZE,
        strides=PATCH_SIZE,
        padding="valid",
    )(inputs)
    x = layers.Reshape((num_patches, PROJECTION_DIM))(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=PROJECTION_DIM)(positions)
    x = x + pos_embed

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM,
            dropout=DROPOUT_RATE,
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[MLP_DIM, PROJECTION_DIM])
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="vit_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_vit(model, train_ds, val_ds, class_weight):
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
