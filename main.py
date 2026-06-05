from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import cnn
import vit


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("Rice_Image_Dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_dataframe(data_dir):
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    file_paths = []
    labels = []

    for class_name in class_names:
        files = sorted((data_dir / class_name).glob("*.jpg"))
        file_paths.extend([str(p) for p in files])
        labels.extend([class_name] * len(files))

    df = pd.DataFrame({"path": file_paths, "label": labels})
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    df["label_id"] = df["label"].map(class_to_index)
    return df, class_names


def split_dataframe(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label_id"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label_id"],
        random_state=SEED,
    )
    return train_df, val_df, test_df


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ]
        )
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def plot_history(history, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].legend()
    plt.show()


def evaluate_model(model, test_ds, class_names, label="Model"):
    test_loss, test_acc = model.evaluate(test_ds)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n{label} classification report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        xticks_rotation=45,
    )
    plt.show()
    return test_loss, test_acc


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError("Rice_Image_Dataset folder not found.")

    df, class_names = build_dataframe(DATA_DIR)
    train_df, val_df, test_df = split_dataframe(df)

    train_ds = make_dataset(train_df["path"].values, train_df["label_id"].values, training=True)
    val_ds = make_dataset(val_df["path"].values, val_df["label_id"].values, training=False)
    test_ds = make_dataset(test_df["path"].values, test_df["label_id"].values, training=False)

    num_classes = len(class_names)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_df["label_id"].values,
    )
    class_weight = dict(enumerate(class_weights))

    Path("models").mkdir(exist_ok=True)

    print("Training CNN...")
    cnn_model = cnn.build_cnn_model(IMG_SIZE, num_classes)
    cnn_history, cnn_train_time = cnn.train_cnn(
        cnn_model, train_ds, val_ds, class_weight, epochs=20
    )
    plot_history(cnn_history, "CNN")
    cnn_test_loss, cnn_test_acc = evaluate_model(cnn_model, test_ds, class_names, "CNN")
    cnn_params = cnn_model.count_params()

    print("Training ViT...")
    vit_model = vit.build_vit_model(IMG_SIZE, num_classes)
    vit_history, vit_train_time = vit.train_vit(
        vit_model, train_ds, val_ds, class_weight, epochs=15
    )
    plot_history(vit_history, "ViT")
    vit_test_loss, vit_test_acc = evaluate_model(vit_model, test_ds, class_names, "ViT")
    vit_params = vit_model.count_params()

    comparison = pd.DataFrame(
        [
            {
                "model": "CNN",
                "params": cnn_params,
                "test_acc": cnn_test_acc,
                "test_loss": cnn_test_loss,
                "train_time_sec": cnn_train_time,
            },
            {
                "model": "ViT",
                "params": vit_params,
                "test_acc": vit_test_acc,
                "test_loss": vit_test_loss,
                "train_time_sec": vit_train_time,
            },
        ]
    )
    print("\nCNN vs ViT comparison")
    print(comparison)


if __name__ == "__main__":
    main()
