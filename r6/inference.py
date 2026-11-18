
from pathlib import Path
import joblib
import numpy as np
import torch

from .common import LABELS, load_baselines, extract_feature_matrix
from .train import TemporalGRU


MIN_FACE_DETECTION_RATE = 0.60


def validate_checkpoint_contract(ckpt):
    labels = ckpt["labels"]
    feature_names = ckpt["feature_names"]

    if labels != LABELS:
        raise ValueError(
            f"Checkpoint labels {labels} do not match "
            f"the R6 project labels {LABELS}."
        )

    if int(ckpt["num_classes"]) != len(LABELS):
        raise ValueError(
            "Checkpoint class count does not match the R6 label contract."
        )

    if int(ckpt["input_size"]) != len(feature_names):
        raise ValueError(
            "Checkpoint input size does not match its feature schema."
        )

    if int(ckpt["window"]) <= 0:
        raise ValueError("Checkpoint window must be positive.")

    temperature = float(ckpt["temperature"])

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Checkpoint temperature must be finite and positive."
        )


class R6Predictor:
    def __init__(self, artifact_dir, baseline_csv, device=None):
        self.artifact_dir = Path(artifact_dir)

        ckpt = torch.load(
            self.artifact_dir / "r6_gru.pt",
            map_location="cpu",
            weights_only=False
        )

        validate_checkpoint_contract(ckpt)

        self.labels = ckpt["labels"]
        self.window = int(ckpt["window"])
        self.temperature = float(ckpt["temperature"])
        self.feature_names = ckpt["feature_names"]

        self.scaler = joblib.load(
            self.artifact_dir / "r6_preprocessing.joblib"
        )

        if len(self.feature_names) != self.scaler.n_features_in_:
            raise ValueError(
                "Checkpoint feature count does not match scaler."
            )

        self.baselines = load_baselines(baseline_csv)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = TemporalGRU(
            ckpt["input_size"],
            ckpt["hidden_size"],
            ckpt["num_classes"]
        )

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    def predict_npz(self, npz_path, subject_id):
        sid = str(subject_id)

        if sid not in self.baselines.index:
            raise ValueError(
                f"No neutral-pose baseline for subject {sid}"
            )

        features, names, detected = extract_feature_matrix(
            npz_path,
            self.baselines.loc[sid]
        )

        if names != self.feature_names:
            raise ValueError(
                "Feature schema differs from training. "
                "Use the same R3/R4/R7 feature version."
            )

        if len(features) < self.window:
            raise ValueError(
                f"Need at least {self.window} frames; got {len(features)}"
            )

        x = features[-self.window:]
        detection_rate = float(np.mean(detected[-self.window:]))

        if detection_rate < MIN_FACE_DETECTION_RATE:
            raise ValueError(
                f"Insufficient face detection: {detection_rate:.1%}. "
                f"Minimum required: {MIN_FACE_DETECTION_RATE:.0%}."
            )

        x = self.scaler.transform(x).astype(np.float32)

        with torch.no_grad():
            logits = self.model(
                torch.tensor(
                    x[None],
                    dtype=torch.float32,
                    device=self.device
                )
            )

            probs = torch.softmax(
                logits / self.temperature,
                dim=1
            )[0].cpu().numpy()

        return {
            "label": self.labels[int(probs.argmax())],
            "probabilities": {
                k: float(v) for k, v in zip(self.labels, probs)
            },
            "face_detection_rate": detection_rate
        }
