from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_vit_model(
    img_size,
    num_classes,
    patch_size=16,
    projection_dim=64,
    num_heads=4,
    transformer_layers=6,
    mlp_dim=128,
    dropout_rate=0.1,
):
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    inputs = layers.Input(shape=(*img_size, 3))
    patches = layers.Conv2D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    x = layers.Reshape((num_patches, projection_dim))(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + pos_embed

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout_rate,
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[mlp_dim, projection_dim], dropout_rate=dropout_rate)
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="vit_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_vit(model, train_ds, val_ds, class_weight, epochs=15, model_path="models/rice_vit.keras"):
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]

    start_time = time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    train_time = time() - start_time
    return history, train_time
