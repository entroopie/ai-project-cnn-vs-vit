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


class RunWriter:
    """
    Singleton-style context object for a single training run.
    Holds run_id, datetime, results_dir and all output/plotting methods.

    Usage:
        rw = RunWriter(results_dir=Path("results"))
        rw.save_training_history(history, "CNN")
        rw.plot_history(history, "CNN")
        loss, acc = rw.evaluate_model(model, test_ds, class_names, "CNN")
        rw.plot_comparison(comparison_df)
        rw.write_run_index(success=True)
    """

    def __init__(self, results_dir: Path):
        """
        Input:  results_dir: Path
        Output: RunWriter instance with frozen run_id, datetime, and start time
        """
        self.run_id = uuid.uuid4().hex[:8]
        self.run_dt = datetime.now(timezone.utc).isoformat()
        self._run_start = time()
        self.results_dir = results_dir
        results_dir.mkdir(exist_ok=True)
        print(f"Run ID: {self.run_id}")

    def _p(self, label: str) -> str:
        """Return the standard file prefix for this run and label."""
        return f"{self.run_id}_{label.lower()}"

    def save_training_history(self, history, label: str):
        """
        Input:  history: keras.callbacks.History, label: str
        Output: None  # writes {run_id}_{label}_training_history.csv
        """
        pd.DataFrame(history.history).to_csv(
            self.results_dir / f"{self._p(label)}_training_history.csv", index=False
        )

    def plot_history(self, history, title_prefix: str):
        """
        Input:  history: keras.callbacks.History, title_prefix: str
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
        plt.savefig(self.results_dir / f"{self._p(title_prefix)}_learning_curves.png", bbox_inches="tight")
        plt.show()

    def evaluate_model(self, model, test_ds, class_names: list, label: str):
        """
        Input:  model: keras.Model, test_ds: tf.data.Dataset, class_names: list[str], label: str
        Output: (test_loss: float, test_acc: float)
        """
        prefix = self._p(label)
        test_loss, test_acc = model.evaluate(test_ds)
        y_true = np.concatenate([y.numpy() for _, y in test_ds])
        y_pred = np.argmax(model.predict(test_ds), axis=1)

        print(f"\n{label} classification report")
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=class_names))
        pd.DataFrame(report_dict).transpose().to_csv(
            self.results_dir / f"{prefix}_classification_report.csv"
        )

        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=class_names, cmap="Blues", xticks_rotation=45
        )
        plt.savefig(self.results_dir / f"{prefix}_confusion_matrix.png", bbox_inches="tight")
        plt.show()
        return test_loss, test_acc

    def plot_comparison(self, comparison):
        """
        Input:  comparison: DataFrame{model, params, test_acc, test_loss, train_time_sec}
        Output: None  # saves CSV + PNG and renders figure
        """
        comparison.to_csv(self.results_dir / f"{self.run_id}_cnn_vs_vit_comparison.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        comparison.plot(x="model", y="test_acc", kind="bar", ax=axes[0], title="Test Accuracy", legend=False, color=["blue", "orange"])
        comparison.plot(x="model", y="params", kind="bar", ax=axes[1], title="Parameter Count", legend=False, color=["blue", "orange"])
        comparison.plot(x="model", y="train_time_sec", kind="bar", ax=axes[2], title="Training Time (s)", legend=False, color=["blue", "orange"])
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{self.run_id}_cnn_vs_vit_comparison_plots.png")
        plt.show()

    def write_run_index(self, success: bool):
        """
        Input:  success: bool
        Output: None  # appends run entry to results_index.json
        """
        rid = self.run_id
        tagged_files = [
            f"{rid}_cnn_training_history.csv",
            f"{rid}_cnn_learning_curves.png",
            f"{rid}_cnn_classification_report.csv",
            f"{rid}_cnn_confusion_matrix.png",
            f"{rid}_vit_training_history.csv",
            f"{rid}_vit_learning_curves.png",
            f"{rid}_vit_classification_report.csv",
            f"{rid}_vit_confusion_matrix.png",
            f"{rid}_cnn_vs_vit_comparison.csv",
            f"{rid}_cnn_vs_vit_comparison_plots.png",
        ]
        entry = {
            "run_id": rid,
            "datetime": self.run_dt,
            "duration_sec": round(time() - self._run_start, 2),
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
        index_path = self.results_dir / "results_index.json"
        runs = json.loads(index_path.read_text()) if index_path.exists() else []
        runs.append(entry)
        index_path.write_text(json.dumps(runs, indent=2))
        print(f"\nRun index updated: results/results_index.json (run_id={rid}, success={success})")

