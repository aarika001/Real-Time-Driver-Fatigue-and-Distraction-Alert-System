from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from r6.common import LABELS, BASE_FEATURES
from r6.train import TemporalGRU
from r6.inference import R6Predictor


def create_test_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    baseline_path = tmp_path / "baselines.csv"
    pd.DataFrame([{
        "subject_id": "TEST_01",
        "mean_pitch": 0.0,
        "mean_yaw": 0.0,
        "std_pitch": 1.0,
    }]).to_csv(baseline_path, index=False)

    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(42).normal(
        size=(200, len(BASE_FEATURES))
    ))
    joblib.dump(scaler, artifact_dir / "r6_preprocessing.joblib")

    model = TemporalGRU(
        input_size=len(BASE_FEATURES),
        hidden_size=64,
        num_classes=3,
    )

    checkpoint = {
        "labels": LABELS,
        "feature_names": BASE_FEATURES,
        "input_size": len(BASE_FEATURES),
        "hidden_size": 64,
        "num_classes": 3,
        "window": 75,
        "temperature": 1.0,
        "state_dict": model.state_dict(),
    }

    torch.save(checkpoint, artifact_dir / "r6_gru.pt")

    return artifact_dir, baseline_path


def create_test_recording(tmp_path, frames=100):
    rng = np.random.default_rng(123)

    npz_path = tmp_path / "test_recording.npz"

    np.savez_compressed(
        npz_path,
        mar=rng.uniform(0.1, 0.5, frames).astype(np.float32),
        head_pose=rng.normal(0, 2, (frames, 3)).astype(np.float32),
        quality=rng.uniform(0, 1, (frames, 4)).astype(np.float32),
        face_detected=np.ones(frames, dtype=np.float32),
    )

    return npz_path


def test_full_r6_inference_pipeline(tmp_path):
    artifact_dir, baseline_path = create_test_artifacts(tmp_path)
    recording = create_test_recording(tmp_path)

    predictor = R6Predictor(
        artifact_dir=artifact_dir,
        baseline_csv=baseline_path,
        device="cpu",
    )

    result = predictor.predict_npz(recording, "TEST_01")

    assert result["label"] in LABELS
    assert set(result["probabilities"].keys()) == set(LABELS)

    probabilities = np.array(list(result["probabilities"].values()))

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)

    assert result["face_detection_rate"] == pytest.approx(1.0)


def test_inference_rejects_missing_subject_baseline(tmp_path):
    artifact_dir, baseline_path = create_test_artifacts(tmp_path)
    recording = create_test_recording(tmp_path)

    predictor = R6Predictor(
        artifact_dir=artifact_dir,
        baseline_csv=baseline_path,
        device="cpu",
    )

    with pytest.raises(ValueError, match="No neutral-pose baseline"):
        predictor.predict_npz(recording, "UNKNOWN_SUBJECT")


def test_inference_rejects_short_recording(tmp_path):
    artifact_dir, baseline_path = create_test_artifacts(tmp_path)
    recording = create_test_recording(tmp_path, frames=50)

    predictor = R6Predictor(
        artifact_dir=artifact_dir,
        baseline_csv=baseline_path,
        device="cpu",
    )

    with pytest.raises(ValueError, match="Need at least"):
        predictor.predict_npz(recording, "TEST_01")
