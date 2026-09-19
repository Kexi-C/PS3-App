"""Train the Door model once on Train.csv and save it for the app.

Usage (run from the app directory):
    python subsystems/door/train_model.py --train <Train.csv> --answer <Train_Segments_Answer.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE / "code"
DEFAULT_MODEL_PATH = HERE / "model" / "door_model.joblib"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import door_standalone as ds  # noqa: E402  (imported by module name so the pickle can be reloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to Door Train.csv")
    parser.add_argument("--answer", required=True, help="Path to Door Train_Segments_Answer.csv")
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="Output .joblib path")
    args = parser.parse_args()

    segments = ds.split_segments(ds.load_stream(args.train))
    labels = pd.read_csv(args.answer)["status"].tolist()
    if len(segments) != len(labels):
        raise SystemExit(
            f"Segmentation mismatch: {len(segments)} segments found, {len(labels)} in the answer file."
        )

    model = ds.DoorModel().fit(segments, labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    n_abnormal = sum(label == ds.ABN for label in labels)
    print(f"Trained on {len(labels)} segments ({n_abnormal} abnormal). Model saved to {out_path}")


if __name__ == "__main__":
    main()
