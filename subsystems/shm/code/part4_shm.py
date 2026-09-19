"""Part 4 SHM: framework-independent inference, ready to import into a shared app.

Dependency: numpy (pip install numpy). No server, UI, training pipeline, model
download, external service, GPU, or other source file is required.

Public API
----------
predict_shm(source, file_id=None, *, include_details=False) -> dict
    source: path, CSV bytes, or a binary upload/file object (Streamlit UploadedFile,
    Flask FileStorage, FastAPI UploadFile.file, etc.). Async upload objects should
    be read by the caller first: data = await upload.read().
    Bytes without a filename require file_id. File-object position is restored
    when seek/tell is available. A getvalue() method is preferred for uploads.
    Default result: {"file_id": "test01.csv", "prediction": float}.
    include_details=True adds sample statistics, warnings and chart-ready data.

predict_shm_batch(files, *, include_details=False) -> list[dict]
    files: mapping filename -> source, or iterable of (filename, source) pairs.
    All files must have unique basenames. Invalid input raises ValueError; this
    function does not silently produce partial submission output.

export_shm_csv(results) -> bytes
    UTF-8 CSV with exactly file_id,prediction columns, sorted by source filename.
    Accepts both minimal and detailed results.

Example for a unified application
--------------------------------
    from part4_shm import predict_shm, predict_shm_batch, export_shm_csv
    result = predict_shm(upload_bytes, "test01.csv", include_details=True)
    damage = result["prediction"]
    results = predict_shm_batch([(f.name, f) for f in uploaded_files])
    download_bytes = export_shm_csv(results)

Input: one numeric column, no header, at least 3 samples. Original stress units
and order are preserved. No smoothing, detrending, or per-file normalization.
Training records contained 581120 samples. Units/sample rate were unspecified.
The range is divided by 2 to obtain amplitude; endpoint half cycles are retained.

Frozen final model: D = 1.3602318550841181e-09 * sum(count * amplitude**5).
Calibrated on the supplied 64 training files, using MAPE-based scale/exponent
fitting. Nested internal validation MAPE: 0.026982602010649573 (about 2.70%).
That is not the official test score. No test labels were used. The calibrated
constant is not a published material constant; no remaining-life claim is made.
The module matches the previously delivered Python inference engine.

Chart fields when include_details=True
-------------------------------------
waveform: [[sample_index, stress], ...], an extrema-preserving display envelope;
          prediction always uses the full signal.
bins: [{low, high, count, damage}, ...], 16 stress-amplitude intervals.
cycle_count includes half cycles. Sum(bin.damage) + empirical_offset equals
prediction to floating-point tolerance. warnings are data-range notices, not
anomaly diagnoses. The prediction is not clipped to [0,1].
"""
import csv
import io
import os
from collections.abc import Mapping
from pathlib import Path
import numpy as np

__all__ = ["predict_shm", "predict_shm_batch", "export_shm_csv"]

_MODEL = {
    "schema_version": 1,
    "kind": "power_mape",
    "m": 5.0,
    "a": 1.3602318550841181e-09,
    "b": 0.0,
    "training_samples_per_file": 581120,
    "training_max_amplitude_range": [18.710071499999998, 47.6414855],
    "nested_validation_mape": 0.026982602010649573,
}

def _file_id(value):
    if value is None:
        raise ValueError("file_id is required for raw CSV bytes.")
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    if not name or not name.lower().endswith(".csv") or any(c in name for c in "\r\n\x00"):
        raise ValueError("file_id must be a CSV source filename, including its extension.")
    return name

def _source_bytes(source, file_id):
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        return path.read_bytes(), _file_id(file_id if file_id is not None else path.name)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), _file_id(file_id)
    inferred = getattr(source, "filename", None) or getattr(source, "name", None)
    name = _file_id(file_id if file_id is not None else inferred)
    if callable(getattr(source, "getvalue", None)):
        data = source.getvalue()
    elif callable(getattr(source, "read", None)):
        position = None
        try:
            position = source.tell()
            source.seek(0)
        except (AttributeError, OSError, io.UnsupportedOperation):
            position = None
        try:
            data = source.read()
        finally:
            if position is not None:
                source.seek(position)
    else:
        raise TypeError("source must be a path, CSV bytes, or a readable upload object.")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Use a binary file/upload object. Await async upload.read() before calling predict_shm.")
    return bytes(data), name

def predict_shm(source, file_id=None, *, include_details=False):
    """Predict a single SHM record; output contains only JSON-serializable values."""
    raw, name = _source_bytes(source, file_id)
    if include_details:
        return analyze(raw, name, _MODEL)
    x = read_stress(raw)
    return {"file_id": name, "prediction": predict_cycles(rainflow(x), _MODEL)}

def predict_shm_batch(files, *, include_details=False):
    """Process named uploads; reject duplicate source filenames before inference."""
    entries = list(files.items() if isinstance(files, Mapping) else files)
    if not entries:
        raise ValueError("No SHM files supplied.")
    names = [_file_id(entry[0]) for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate CSV basenames; preserve unique source filenames.")
    results = []
    for name, (_, source) in zip(names, entries):
        try:
            results.append(predict_shm(source, name, include_details=include_details))
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
    return results

def export_shm_csv(results):
    """Return competition-format UTF-8 CSV bytes; ignore optional detail fields."""
    rows = []
    for result in results:
        name = _file_id(result["file_id"])
        value = float(result["prediction"])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name}: prediction must be finite and nonnegative.")
        rows.append((name, value))
    if not rows:
        raise ValueError("No SHM predictions to export.")
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError("Duplicate source filenames cannot be exported.")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["file_id", "prediction"])
    for name, value in sorted(rows):
        writer.writerow([name, format(value, ".12g")])
    return output.getvalue().encode("utf-8")

def read_stress(data):
    """Read a headerless, single-column CSV; never silently skip a data row."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    try:
        text = data.decode('utf-8-sig')
        if any(not line.strip() for line in text.splitlines()):
            raise ValueError('Blank rows are not supported; check the source file.')
        x = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2, comments=None)
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Expected a headerless CSV with one finite numeric value per row. ' + str(exc)) from exc
    if x.ndim != 2 or x.shape[1] != 1 or x.shape[0] < 3:
        raise ValueError('Expected one column and at least three samples.')
    x = x[:, 0]
    if not np.isfinite(x).all():
        raise ValueError('NaN and infinite values are not supported.')
    return x

def rainflow(x):
    """Return columns [amplitude, count, mean] using stack rainflow counting.

    End residues are counted as half cycles. No smoothing, hysteresis threshold,
    detrending or amplitude normalization is applied. Amplitude = range / 2.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Rainflow requires a finite one-dimensional array.')
    x = x[np.r_[True, np.diff(x) != 0]] if x.size else x
    if x.size < 2:
        return np.empty((0, 3))
    d = np.diff(x)
    rev = x[np.r_[True, np.signbit(d[:-1]) != np.signbit(d[1:]), True]]
    stack, cycles = [], []
    for p in rev:
        stack.append(float(p))
        while len(stack) >= 3:
            old = abs(stack[-2] - stack[-3])
            new = abs(stack[-1] - stack[-2])
            if new < old:
                break
            avg = (stack[-3] + stack[-2]) / 2
            if len(stack) == 3:
                cycles.append((old / 2, .5, avg))
                stack.pop(0)
            else:
                cycles.append((old / 2, 1., avg))
                last = stack.pop()
                stack.pop(); stack.pop(); stack.append(last)
    cycles.extend((abs(a-b)/2, .5, (a+b)/2) for a, b in zip(stack[:-1], stack[1:]))
    return np.asarray(cycles, dtype=float).reshape(-1, 3)

def cycle_moment(cycles, m):
    with np.errstate(over='raise', invalid='raise'):
        try:
            return float(np.sum(cycles[:, 1] * cycles[:, 0] ** m))
        except FloatingPointError as exc:
            raise ValueError('Stress magnitude is too large for the fitted model.') from exc

def predict_cycles(cycles, model):
    value = model['b'] + model['a'] * cycle_moment(cycles, model['m'])
    if not np.isfinite(value) or value < 0:
        raise ValueError('Prediction is not finite and nonnegative.')
    return float(value)

def waveform_envelope(x, bins=600):
    # Preserve local extrema in the plot, instead of a stride that can miss spikes.
    out = []
    edges = np.linspace(0, len(x), min(bins, len(x)) + 1, dtype=int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        block = x[lo:hi]
        ids = sorted(set([lo + int(np.argmin(block)), lo + int(np.argmax(block))]))
        out.extend([[int(i), float(x[i])] for i in ids])
    return out

def analyze(data, filename, model):
    x = read_stress(data)
    cycles = rainflow(x)
    pred = predict_cycles(cycles, model)
    weights = model['a'] * cycles[:, 1] * cycles[:, 0] ** model['m']
    top = float(cycles[:, 0].max()) if len(cycles) else 1.
    edges = np.linspace(0., max(top, 1e-12), 17)
    counts = np.histogram(cycles[:, 0], bins=edges, weights=cycles[:, 1])[0]
    damage = np.histogram(cycles[:, 0], bins=edges, weights=weights)[0]
    warnings = []
    if len(x) != model['training_samples_per_file']:
        warnings.append('Record length differs from training files; performance at this length is unvalidated.')
    maxamp = float(cycles[:, 0].max()) if len(cycles) else 0.
    low, high = model['training_max_amplitude_range']
    if not low <= maxamp <= high:
        warnings.append('Maximum cycle amplitude is outside the training range.')
    if len(cycles) == 0:
        warnings.append('Constant signal: no stress cycles were detected.')
    return dict(file_id=filename, prediction=pred, samples=int(x.size),
                mean=float(x.mean()), std=float(x.std()),
                cycle_count=float(cycles[:, 1].sum()), max_amplitude=maxamp,
                empirical_offset=float(model['b']), warnings=warnings,
                waveform=waveform_envelope(x),
                bins=[dict(low=float(edges[i]), high=float(edges[i+1]),
                           count=float(counts[i]), damage=float(damage[i])) for i in range(16)])
