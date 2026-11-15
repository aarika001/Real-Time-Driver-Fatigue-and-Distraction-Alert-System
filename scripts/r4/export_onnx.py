"""Export the trained eye CNN to ONNX for fast CPU inference (R4, Milestone 6).

The exported graph takes (B, 1, 64, 64) float32 crops in [0, 1] and returns
(B,) calibrated P(closed): temperature and sigmoid are baked in. After export
the script checks that ONNX Runtime and PyTorch agree on random inputs.

Example:
    python scripts/r4/export_onnx.py --checkpoint runs/eye_cnn/best.pt --out runs/eye_cnn/eye_cnn.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from cues.eye_model import load_checkpoint  # noqa: E402


class _ProbabilityHead(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model.calibrated_logits(x))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    model, _ = load_checkpoint(args.checkpoint)
    wrapper = _ProbabilityHead(model).eval()
    dummy = torch.rand(2, 1, 64, 64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["crops"], output_names=["p_closed"],
                  dynamic_axes={"crops": {0: "batch"}, "p_closed": {0: "batch"}},
                  opset_version=args.opset, do_constant_folding=True)
    try:   # newer PyTorch defaults to the dynamo exporter; the classic one is simpler and robust here
        torch.onnx.export(wrapper, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:   # older PyTorch without the `dynamo` argument
        torch.onnx.export(wrapper, dummy, str(out), **kwargs)
    print(f"Exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    try:
        from cues.predictors import OnnxEyePredictor
    except ImportError:
        print("onnxruntime not installed; skipping the consistency check")
        return 0
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, size=(8, 64, 64), dtype=np.uint8)
    with torch.inference_mode():
        ref = wrapper(torch.from_numpy(crops).float().div(255).unsqueeze(1)).numpy()
    got = OnnxEyePredictor(out).predict_crops(crops)
    diff = float(np.abs(ref - got).max())
    print(f"Max |PyTorch - ONNX| on 8 random crops: {diff:.2e}")
    if diff > 1e-3:
        print("ERROR: ONNX output differs from PyTorch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
