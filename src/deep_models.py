from __future__ import annotations

import time
import numpy as np

from .utils import ts_print

def to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def train_keras_dnn(X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 256):
    import tensorflow as tf

    X_train = to_dense(X_train)
    X_val = to_dense(X_val)

    ts_print(f"Keras DNN data shapes: X_train={X_train.shape}, X_val={X_val.shape}")

    total_batches = int(np.ceil(len(X_train) / batch_size))
    progress_every = max(1, total_batches // 10)

    class BatchProgressLogger(tf.keras.callbacks.Callback):
        def __init__(self, total_batches: int, every: int):
            self.total_batches = total_batches
            self.every = every
            self.epoch_start = None

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.time()
            ts_print(f"Epoch {epoch + 1}/{epochs} started.")

        def on_train_batch_end(self, batch, logs=None):
            if (batch + 1) % self.every == 0 or (batch + 1) == self.total_batches:
                elapsed = time.time() - self.epoch_start if self.epoch_start else 0.0
                loss = logs.get("loss") if logs else None
                loss_str = f"{loss:.4f}" if loss is not None else "n/a"
                ts_print(
                    f"  batch {batch + 1}/{self.total_batches} - loss={loss_str} - elapsed={elapsed:.1f}s"
                )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")
            val_rmse = logs.get("val_rmse") or logs.get("rmse")
            val_mae = logs.get("val_mae") or logs.get("mae")
            parts = []
            if val_loss is not None:
                parts.append(f"val_loss={val_loss:.4f}")
            if val_rmse is not None:
                parts.append(f"val_rmse={val_rmse:.4f}")
            if val_mae is not None:
                parts.append(f"val_mae={val_mae:.4f}")
            metrics = " - " + " ".join(parts) if parts else ""
            ts_print(f"Epoch {epoch + 1}/{epochs} finished{metrics}.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        BatchProgressLogger(total_batches=total_batches, every=progress_every),
    ]

    ts_print(f"Starting Keras DNN training: epochs={epochs}, batch_size={batch_size}")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    return model


def predict_keras_dnn(model, X):
    X = to_dense(X)
    return model.predict(X, verbose=0).reshape(-1)
