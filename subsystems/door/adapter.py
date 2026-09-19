"""Streamlit adapter for the Door inference program (door_standalone.py).

The prediction logic of door_standalone.py is used unchanged; this module only
loads the saved model, runs inference on one continuous stream CSV, and shapes
the output for the app.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE / "code"
MODEL_PATH = HERE / "model" / "door_model.joblib"
SUBMISSION_COLUMNS = ["start_time", "end_time", "prediction"]


def _module():
    """Import door_standalone by its own module name (required to unpickle DoorModel)."""
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    return importlib.import_module("door_standalone")


@lru_cache(maxsize=1)
def load_model():
    if not CODE_DIR.joinpath("door_standalone.py").is_file():
        raise FileNotFoundError(f"Door predictor not found: {CODE_DIR / 'door_standalone.py'}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Door model not found: {MODEL_PATH}. Run subsystems/door/train_model.py first."
        )
    _module()
    return joblib.load(MODEL_PATH)


def _check_columns(path: Path, ds) -> None:
    header = pd.read_csv(path, nrows=0).columns
    required = ["Datetime", ds.COL_I, ds.COL_V, ds.COL_E, ds.COL_P, ds.COL_OPENING]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"The uploaded file is not a Door data stream. Missing columns: {missing}")


def predict(input_path: Path) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Segment the stream and classify every cycle.

    Returns the detailed result table (one row per detected cycle) and the
    list of raw segments (used for the per-cycle anomaly timeline).
    """
    ds = _module()
    model = load_model()
    input_path = Path(input_path)
    _check_columns(input_path, ds)

    if not ds.split_segments(ds.load_stream(input_path)):
        raise ValueError("No door open/close cycle was detected in the uploaded file.")

    detail, segments = ds.predict_stream(input_path, model)
    detail.insert(0, "segment", range(1, len(detail) + 1))
    return detail, segments


def submission_table(detail: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly the three columns required by the Door submission format."""
    missing = [column for column in SUBMISSION_COLUMNS if column not in detail.columns]
    if missing:
        raise ValueError(f"Door prediction is missing required columns: {missing}")
    return detail[SUBMISSION_COLUMNS].copy()


def timeline(segment: pd.DataFrame, step_ms: int = 20) -> pd.DataFrame:
    """Per-cycle anomaly trace: channel z-scores against the normal template."""
    return _module().anomaly_timeline(segment, load_model(), step_ms=step_ms)
