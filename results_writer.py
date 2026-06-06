import json
from datetime import datetime, timezone
from pathlib import Path
from time import time
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import cnn
import vit


def new_run():
    """
    Output: (run_id: str, run_dt: str, run_start: float)
    """
    run_id = uuid.uuid4().hex[:8]
    run_dt = datetime.now(timezone.utc).isoformat()
    run_start = time()
    return run_id, run_dt, run_start


def save_training_history(history, label, results_dir, run_id):
    """
    Input:  history: keras.callbacks.History, label: str, results_dir: Path, run_id: str
    Output: None  # writes CSV to results_dir
    """
    pd.DataFrame(history.history).to_csv(
        results_dir / f"{run_id}_{label.lower()}_training_history.csv", index=False
    )


def plot_history(history, title_prefix, results_dir, run_id):
    """
    Input:  history: keras.callbacks.History, title_prefix: str, results_dir: Path, run_id: str
    Output: None  # saves PNG and renders figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].legend()

    plt.savefig(results_dir / f"{run_id}_{title_prefix.lower()}_learning_curves.png", bbox_inches="tight")
    plt.show()


def evaluate_model(model, test_ds, class_names, label, results_dir, run_id):
    """
    Input:  model: keras.Model, test_ds: tf.data.Dataset, class_names: list[str], label: str, results_dir: Path, run_id: str
    Output: (test_loss: float, test_acc: float)
    """
    prefix = f"{run_id}_{label.lower()}"
    test_loss, test_acc = model.evaluate(test_ds)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n{label} classification report")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    pd.DataFrame(report_dict).transpose().to_csv(results_dir / f"{prefix}_classification_report.csv")

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        xticks_rotation=45,
    )
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", bbox_inches="tight")
    plt.show()
    return test_loss, test_acc


def plot_comparison(comparison, results_dir, run_id):
    """
    Input:  comparison: DataFrame{model, params, test_acc, test_loss, train_time_sec}, results_dir: Path, run_id: str
    Output: None  # saves PNG and renders figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    comparison.plot(x="model", y="test_acc", kind="bar", ax=axes[0], title="Test Accuracy", legend=False, color=["blue", "orange"])
    comparison.plot(x="model", y="params", kind="bar", ax=axes[1], title="Parameter Count", legend=False, color=["blue", "orange"])
    comparison.plot(x="model", y="train_time_sec", kind="bar", ax=axes[2], title="Training Time (s)", legend=False, color=["blue", "orange"])
    plt.tight_layout()
    plt.savefig(results_dir / f"{run_id}_cnn_vs_vit_comparison_plots.png")
    plt.show()


def write_run_index(results_dir, run_id, run_dt, run_start, success):
    """
    Input:  results_dir: Path, run_id: str, run_dt: str, run_start: float, success: bool
    Output: None  # appends entry to results_index.json
    """
    tagged_files = [
        f"{run_id}_cnn_training_history.csv",
        f"{run_id}_cnn_learning_curves.png",
        f"{run_id}_cnn_classification_report.csv",
        f"{run_id}_cnn_confusion_matrix.png",
        f"{run_id}_vit_training_history.csv",
        f"{run_id}_vit_learning_curves.png",
        f"{run_id}_vit_classification_report.csv",
        f"{run_id}_vit_confusion_matrix.png",
        f"{run_id}_cnn_vs_vit_comparison.csv",
        f"{run_id}_cnn_vs_vit_comparison_plots.png",
    ]
    entry = {
        "run_id": run_id,
        "datetime": run_dt,
        "duration_sec": round(time() - run_start, 2),
        "success": success,
        "config": {
            "cnn": {
                "learning_rate": cnn.LEARNING_RATE,
                "epochs": cnn.EPOCHS,
                "dropout_conv": cnn.DROPOUT_CONV,
                "dropout_fc": cnn.DROPOUT_FC,
                "dense_units": cnn.DENSE_UNITS,
                "early_stopping_patience": cnn.EARLY_STOPPING_PATIENCE,
                "reduce_lr_factor": cnn.REDUCE_LR_FACTOR,
                "reduce_lr_patience": cnn.REDUCE_LR_PATIENCE,
                "reduce_lr_min_lr": cnn.REDUCE_LR_MIN_LR,
            },
            "vit": {
                "learning_rate": vit.LEARNING_RATE,
                "epochs": vit.EPOCHS,
                "patch_size": vit.PATCH_SIZE,
                "projection_dim": vit.PROJECTION_DIM,
                "num_heads": vit.NUM_HEADS,
                "transformer_layers": vit.TRANSFORMER_LAYERS,
                "mlp_dim": vit.MLP_DIM,
                "dropout_rate": vit.DROPOUT_RATE,
                "early_stopping_patience": vit.EARLY_STOPPING_PATIENCE,
                "reduce_lr_factor": vit.REDUCE_LR_FACTOR,
                "reduce_lr_patience": vit.REDUCE_LR_PATIENCE,
                "reduce_lr_min_lr": vit.REDUCE_LR_MIN_LR,
            },
        },
        "files": tagged_files,
    }
    index_path = results_dir / "results_index.json"
    runs = json.loads(index_path.read_text()) if index_path.exists() else []
    runs.append(entry)
    index_path.write_text(json.dumps(runs, indent=2))
    print(f"\nRun index updated: results/results_index.json (run_id={run_id}, success={success})")
