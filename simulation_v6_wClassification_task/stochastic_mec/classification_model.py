"""Empirical Digit Classification Task (MNIST 0-9) for Split-Computing & ISCC Evaluation.

Bridges the theoretical ISCC framework (feature retention chi, task splitting rho,
sensing SNR gamma^sen, and accuracy weight beta^a) with a real-world Split-CNN
evaluating top-1 digit classification accuracy, end-to-end latency, and energy.

System Specifications & Topology Overview:
  - Topology: 1000m x 1000m ground area with Voronoi cell partitioning.
  - UAVs: 2 UL MEC UAVs (hovering at z in [80, 120]m with 8 GHz server capacity)
    and 1 DL UAV for downlink communications.
  - UEs: 5 to 35 active ground UEs (default: 8 to 10 UEs).
  - Task Workload: Each UE processes a burst/sequence of ~50-100 digits per time
    block (D_raw = 420 kbits, C_raw = 1.0 Gcycles) under heterogeneous sensing
    SNRs (gamma_sen ~ N(10, 3^2) dB), local CPU speeds, and wireless channels.

See note_classification_simulation_setup.md for full architectural details.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import SystemParams

HERE = Path(__file__).resolve().parent
WEIGHTS_FILE = HERE / "mnist_split16_weights.npz"


# ==============================================================================
# Model Architecture & Feature Pruning (BottleNet Split-CNN - 16 Channels)
# ==============================================================================
class SplitCNN:
    """BottleNet-style Split-CNN with on-device feature extraction and UAV-MEC classification.
    
    Architecture:
      - Head (UE): Conv2D(16, 3x3) -> MaxPool(2x2) -> Conv2D(16, 3x3) -> MaxPool(2x2)
                   Output feature tensor shape: (Batch, 7, 7, 16)
      - Channel Retention (chi): Retains top K_chi = max(1, round(16 * chi)) channels.
                   Transmitted feature size: 7 * 7 * K_chi * 8 bits (INT8 quantized).
                   Provides 1x to 16x data reduction compared to raw 28x28 images (6272 bits).
      - Tail (MEC Server): SpatialDropout -> Conv2D(16, 3x3) -> MaxPool -> Dense(64) -> Dense(10, Softmax)
    """

    def __init__(self, weights_path: Path | None = None):
        self.weights_path = weights_path or WEIGHTS_FILE
        self.head_model, self.tail_model, self.full_model = self._build_models()
        self._initialized = False

    def _build_models(self) -> tuple[keras.Model, keras.Model, keras.Model]:
        # Head (On-Device Feature Extractor with 16 Bottleneck Channels)
        head_input = keras.Input(shape=(28, 28, 1), name="image_input")
        x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="head_conv1")(head_input)
        x = layers.MaxPooling2D((2, 2), name="head_pool1")(x)            # (B, 14, 14, 16)
        x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="head_bottleneck")(x)
        head_output = layers.MaxPooling2D((2, 2), name="head_pool2")(x)  # (B, 7, 7, 16)
        head_model = keras.Model(inputs=head_input, outputs=head_output, name="head_model")

        # Tail (UAV-MEC Deep Classifier)
        tail_input = keras.Input(shape=(7, 7, 16), name="feature_input")
        y = layers.SpatialDropout2D(0.15, name="tail_channel_dropout")(tail_input)
        y = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="tail_conv2")(y)
        y = layers.MaxPooling2D((2, 2), name="tail_pool2")(y)            # (B, 3, 3, 16)
        y = layers.Flatten(name="tail_flatten")(y)
        y = layers.Dense(64, activation="relu", name="tail_dense1")(y)
        tail_output = layers.Dense(10, activation="softmax", name="tail_output")(y)
        tail_model = keras.Model(inputs=tail_input, outputs=tail_output, name="tail_model")

        # Combined Full Model (For All-Local or Raw Offload execution)
        full_output = tail_model(head_model(head_input))
        full_model = keras.Model(inputs=head_input, outputs=full_output, name="full_model")

        full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return head_model, tail_model, full_model

    def train_or_load(self, force_retrain: bool = False, epochs: int = 3) -> None:
        """Loads cached weights or quickly trains on MNIST in ~5 seconds."""
        if self._initialized and not force_retrain:
            return

        if not force_retrain and self.weights_path.exists():
            try:
                data = np.load(self.weights_path)
                weights = [data[f"w_{i}"] for i in range(len(data.files))]
                self.full_model.set_weights(weights)
                self._initialized = True
                return
            except Exception:
                pass

        # Train on MNIST
        try:
            (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
            x_train = (x_train.astype(np.float32) / 255.0)[..., None]
            x_val = (x_val.astype(np.float32) / 255.0)[..., None]
        except Exception:
            # Synthetic fallback if offline
            rng = np.random.default_rng(42)
            x_train = rng.uniform(0.0, 1.0, (2000, 28, 28, 1)).astype(np.float32)
            y_train = rng.integers(0, 10, (2000,)).astype(np.int32)
            x_val = rng.uniform(0.0, 1.0, (500, 28, 28, 1)).astype(np.float32)
            y_val = rng.integers(0, 10, (500,)).astype(np.int32)

        self.full_model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=0,
        )

        try:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_dict = {f"w_{i}": w for i, w in enumerate(self.full_model.get_weights())}
            np.savez_compressed(self.weights_path, **weights_dict)
        except Exception:
            pass

        self._initialized = True


# ==============================================================================
# Digit Classification Benchmark Task
# ==============================================================================
@dataclass
class ClassificationResult:
    scheme: str
    chi: float
    rho: float
    accuracy: float
    delay_ms: float
    energy_mj: float
    transmitted_kbits: float
    utility: float


class DigitClassificationTask:
    """Manages empirical evaluation of MNIST classification under ISCC parameters."""

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.default_rng(rng_seed)
        self.cnn = SplitCNN()
        self.cnn.train_or_load()
        self._load_test_data()

        # Profiling parameters (FLOPs / CPU cycles estimation)
        # Head (16 channels): Conv1 (113k) + Conv2 (452k) ~ 5.65e5 MACs
        # Tail (16 channels): Conv3 (113k) + Dense1 (9k) + Dense2 (640) ~ 1.23e5 MACs
        self.head_macs = 5.65e5
        self.tail_macs = 1.23e5
        self.full_macs = self.head_macs + self.tail_macs
        self.cycles_per_mac = 4.0   # Embedded ARM / DSP processor cycles per MAC

    def _load_test_data(self) -> None:
        try:
            _, (x_test, y_test) = keras.datasets.mnist.load_data()
            self.x_test = (x_test.astype(np.float32) / 255.0)[..., None]
            self.y_test = y_test.astype(np.int32)
        except Exception:
            self.x_test = self.rng.uniform(0.0, 1.0, (1000, 28, 28, 1)).astype(np.float32)
            self.y_test = self.rng.integers(0, 10, (1000,)).astype(np.int32)

    def get_sample_batch(self, n_samples: int = 1000, snr_db: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """Samples test images degraded by sensing SNR."""
        n = min(n_samples, len(self.x_test))
        idx = self.rng.choice(len(self.x_test), n, replace=False)
        x_sub = self.x_test[idx].copy()
        y_sub = self.y_test[idx].copy()

        # Inject sensing noise: SNR = 10^(snr_db / 10)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_std = 1.0 / np.sqrt(max(snr_lin, 1e-6))
        noise = self.rng.normal(0.0, noise_std * 0.15, size=x_sub.shape).astype(np.float32)
        x_noisy = np.clip(x_sub + noise, 0.0, 1.0)
        return x_noisy, y_sub

    def eval_top1_accuracy(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        chi: float = 1.0,
        is_split: bool = True,
    ) -> float:
        """Evaluates empirical Top-1 accuracy with feature channel retention chi in [0, 1]."""
        if not is_split or chi >= 0.999:
            preds = self.cnn.full_model.predict(x_test, verbose=0, batch_size=256)
            acc = float(np.mean(np.argmax(preds, axis=-1) == y_test))
            return acc

        # Split execution: Head -> Prune channels -> Tail
        feats = self.cnn.head_model.predict(x_test, verbose=0, batch_size=256) # (B, 7, 7, 8)
        n_channels = feats.shape[-1]
        keep_k = max(1, int(np.round(n_channels * chi)))

        # Retain top-k channels based on feature magnitude (L1-norm channel pruning)
        channel_importance = np.mean(np.abs(feats), axis=(0, 1, 2))
        top_k = np.argsort(channel_importance)[::-1][:keep_k]

        feats_pruned = np.zeros_like(feats)
        feats_pruned[..., top_k] = feats[..., top_k]

        preds = self.cnn.tail_model.predict(feats_pruned, verbose=0, batch_size=256)
        acc = float(np.mean(np.argmax(preds, axis=-1) == y_test))
        return acc

    def simulate_user_inference(
        self,
        scheme: str,
        chi: float,
        rho: float,
        rate_bps: float,
        snr_db: float = 10.0,
        p_tx: float = 0.20,
        f_loc: float = 0.8e9,
        f_srv: float = 8.0e9,
        params: SystemParams | None = None,
        n_samples: int = 500,
        batch_size: int = 128,
    ) -> ClassificationResult:
        """Computes empirical accuracy, latency, energy, and utility for an inference task (batch of frames)."""
        p = params or SystemParams()
        x_batch, y_batch = self.get_sample_batch(n_samples=n_samples, snr_db=snr_db)

        # Sensing overhead (Lemma 1 & Sec. II-C)
        t_sen = p.sensing_time
        e_sen = p.sensing_power * t_sen

        c_head = batch_size * (self.head_macs * self.cycles_per_mac)
        c_tail = batch_size * (self.tail_macs * self.cycles_per_mac)
        c_full = batch_size * (self.full_macs * self.cycles_per_mac)

        # -------------------------------------------------------------
        # Scheme 1: All-Local (ALCA / No Split, rho = 0, chi = 1)
        # -------------------------------------------------------------
        if scheme.upper() in ("ALCA", "LOCAL", "ALL-LOCAL"):
            acc = self.eval_top1_accuracy(x_batch, y_batch, chi=1.0, is_split=False)
            t_cmp = c_full / f_loc
            e_cmp = p.energy_coefficient * (f_loc ** 2) * c_full
            t_total = t_sen + t_cmp
            e_total = e_sen + e_cmp
            d_tx_kbits = 0.0

        # -------------------------------------------------------------
        # Scheme 2: Full-Offloading (Raw Transmit / Atomic, rho = 1, chi = 1)
        # -------------------------------------------------------------
        elif scheme.upper() in ("RAW", "ALL-OFFLOAD", "ATOMIC", "RAW-OFFLOAD"):
            acc = self.eval_top1_accuracy(x_batch, y_batch, chi=1.0, is_split=False)
            # Raw 28x28 image (8-bit grayscale) = 6272 bits per frame
            raw_bits = batch_size * (28 * 28 * 8.0)
            t_tx = raw_bits / max(rate_bps, 1e-6)
            e_tx = p_tx * t_tx
            t_srv = c_full / f_srv
            t_total = t_sen + t_tx + t_srv
            e_total = e_sen + e_tx
            d_tx_kbits = raw_bits / 1000.0

        # -------------------------------------------------------------
        # Scheme 3: Proposed ISCC Split Computing (rho, chi)
        # -------------------------------------------------------------
        else:
            acc = self.eval_top1_accuracy(x_batch, y_batch, chi=chi, is_split=True)
            # Retained feature tensor size: 7 * 7 * K_chi * 8 bits (INT8 quantized feature map)
            n_ch = max(1, int(np.round(16 * chi)))
            feat_bits = batch_size * (7 * 7 * n_ch * 8.0)

            # Remote Offload branch (fraction rho): Head on UE + Wireless TX + Tail on UAV
            t_head = rho * (c_head / f_loc)
            t_tx = rho * (feat_bits / max(rate_bps, 1e-6))
            t_tail = rho * (c_tail / f_srv)
            t_ofl = t_head + t_tx + t_tail

            # Local Computing branch (fraction 1 - rho): Full model on UE
            t_loc = (1.0 - rho) * (c_full / f_loc)

            # Concurrent execution (Section II-C: T_n = T_sen + max(T_loc, T_ofl))
            if rho >= 0.999:
                t_cmp = t_ofl
            elif rho <= 0.001:
                t_cmp = t_loc
            else:
                t_cmp = max(t_loc, t_ofl)

            t_total = t_sen + t_cmp

            # Energy consumption
            e_head = rho * p.energy_coefficient * (f_loc ** 2) * c_head
            e_tx = rho * p_tx * (feat_bits / max(rate_bps, 1e-6))
            e_loc = (1.0 - rho) * p.energy_coefficient * (f_loc ** 2) * c_full
            e_total = e_sen + e_head + e_tx + e_loc
            d_tx_kbits = rho * feat_bits / 1000.0

        # Utility calculation (Sec. II-D): U = -beta_t * (T / T_ref) - beta_e * (E / E_ref) + beta_a * Acc
        t_ref = 0.500 * (batch_size / 128.0)  # 500 ms reference deadline for 128-frame batch
        e_ref = 0.500 * (batch_size / 128.0)  # 500 mJ reference energy
        u = (
            - p.beta_time * (t_total / t_ref)
            - p.beta_energy * (e_total / e_ref)
            + p.beta_acc * (acc - p.accuracy_threshold) / (1.0 - p.accuracy_threshold)
        )

        return ClassificationResult(
            scheme=scheme,
            chi=float(chi),
            rho=float(rho),
            accuracy=float(acc),
            delay_ms=float(t_total * 1000.0),
            energy_mj=float(e_total * 1000.0),
            transmitted_kbits=float(d_tx_kbits),
            utility=float(u),
        )
