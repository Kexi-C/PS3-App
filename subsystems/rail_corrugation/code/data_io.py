"""原始数据读取与张量重构。

本模块只负责读取 CSV（或 build_cache.py 生成的 .npy 缓存）并重构为结构化数组，
不做任何信号处理。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def natural_key(path: Path) -> tuple:
    """按文件名中的数字自然排序（Train2 排在 Train10 之前）。"""
    m = re.search(r"(\d+)", Path(path).stem)
    return (0, int(m.group(1)), Path(path).name) if m else (1, 0, Path(path).name)


def list_csv_files(directory: Path) -> list[Path]:
    """列出目录中的全部 CSV 文件，排除隐藏文件，按自然顺序排序。"""
    files = [p for p in Path(directory).glob("*.csv") if not p.name.startswith(".")]
    return sorted(files, key=natural_key)


def read_csv_array(csv_path: Path) -> np.ndarray:
    """读取单个 CSV，返回形状为 (T, 129) 的 float32 数组（表头已去除）。"""
    df = pd.read_csv(csv_path, header=0, dtype=np.float32, engine="c")
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != C.N_COLUMNS:
        raise ValueError(f"{Path(csv_path).name}: 期望 {C.N_COLUMNS} 列，实际形状 {arr.shape}")
    return arr


def raw_cache_path(csv_path: Path) -> Path:
    """返回 CSV 对应的 .npy 缓存路径：RAW_CACHE_DIR/<所在目录名>/<文件名>.npy。"""
    csv_path = Path(csv_path)
    return C.RAW_CACHE_DIR / csv_path.parent.name / f"{csv_path.stem}.npy"


def load_raw(csv_path: Path, use_cache: bool = True) -> np.ndarray:
    """优先以内存映射方式读取 .npy 缓存；缓存不存在或 use_cache=False 时读取 CSV。"""
    if use_cache:
        npy = raw_cache_path(csv_path)
        if npy.exists():
            return np.load(npy, mmap_mode="r")
    return read_csv_array(csv_path)


def split_speed_and_signals(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """拆分转速与信号。

    返回
    ----
    speed   : (T,)            转速脉冲序列（0/1）
    signals : (8, 8, 2, T)    维度依次为 车、位置、通道(振动, 冲击)、时间
    """
    speed = np.asarray(raw[:, C.SPEED_COL], dtype=np.float32)
    sig = np.asarray(raw[:, C.SIGNAL_COL_START:], dtype=np.float32)
    t = sig.shape[0]
    signals = sig.reshape(t, C.N_CARS, C.N_POSITIONS, C.N_CHANNEL_TYPES).transpose(1, 2, 3, 0)
    return speed, np.ascontiguousarray(signals)


def side_view(signals: np.ndarray, side: str) -> np.ndarray:
    """按侧别切片，返回 (8, 4, 2, T)。Side I 对应位置 1/3/5/7，Side II 对应位置 2/4/6/8。"""
    idx = {"Side I": C.SIDE_I_POS_IDX, "Side II": C.SIDE_II_POS_IDX}[side]
    return signals[:, list(idx), :, :]


def load_file(csv_path: Path, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """读取单个文件并返回 (speed, signals)。"""
    return split_speed_and_signals(load_raw(csv_path, use_cache=use_cache))


def load_labels(labels_csv: Path = C.LABELS_CSV) -> pd.DataFrame:
    """读取训练标签，返回列 filename、label、label_id，按文件名自然排序。"""
    df = pd.read_csv(labels_csv, dtype=str)
    df.columns = df.columns.str.strip()
    fcol, lcol = C.LABELS_FILE_COL, C.LABELS_LABEL_COL
    df = df[[fcol, lcol]].apply(lambda s: s.str.strip())

    unknown = set(df[lcol]) - set(C.CLASS_NAMES)
    if unknown:
        raise ValueError(f"标签文件中存在未知类别: {unknown}")
    if df[fcol].duplicated().any():
        raise ValueError("标签文件中存在重复文件名")

    df["label_id"] = df[lcol].map(C.LABEL_TO_ID).astype(int)
    order = sorted(range(len(df)), key=lambda i: natural_key(Path(df[fcol].iloc[i])))
    return df.iloc[order].reset_index(drop=True)


def _self_check() -> None:
    raw = read_csv_array(C.TRAIN_DIR / "Train1.csv")
    speed, signals = split_speed_and_signals(raw)

    # 重构后的张量须与 config.signal_column 的列号公式逐点一致
    for car, pos, ch in [(1, 1, "vibration"), (1, 2, "shock"), (3, 5, "vibration"), (8, 8, "shock")]:
        k = C.CHANNEL_TYPES.index(ch)
        assert np.array_equal(signals[car - 1, pos - 1, k], raw[:, C.signal_column(car, pos, ch)])

    # Side I 切片的第 2 个位置应为位置 3；Side II 切片的第 1 个位置应为位置 2
    s1, s2 = side_view(signals, "Side I"), side_view(signals, "Side II")
    assert s1.shape == (C.N_CARS, 4, C.N_CHANNEL_TYPES, raw.shape[0])
    assert np.array_equal(s1[:, 1], signals[:, 2])
    assert np.array_equal(s2[:, 0], signals[:, 1])

    labels = load_labels()
    train_files = list_csv_files(C.TRAIN_DIR)
    test_files = list_csv_files(C.TEST_DIR)
    assert [p.name for p in train_files] == labels[C.LABELS_FILE_COL].tolist()

    print("self-check passed")
    print("raw shape        :", raw.shape, raw.dtype)
    print("signals shape    :", signals.shape)
    print("side view shape  :", s1.shape)
    print("speed values     :", np.unique(speed))
    print("train / test     :", len(train_files), "/", len(test_files))
    print("first / last     :", train_files[0].name, train_files[-1].name,
          "|", test_files[0].name, test_files[-1].name)
    print(labels[C.LABELS_LABEL_COL].value_counts().to_string())


if __name__ == "__main__":
    _self_check()
