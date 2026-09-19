"""程序：数据预处理与缓存。

输入：Train/ 与 Test/ 中的全部原始 CSV，以及 Train_Labels.csv（只读，不修改）。
输出（写入 CACHE_DIR，位于仓库之外）：
  raw_npy/Train/*.npy, raw_npy/Test/*.npy   原始信号的 float32 二进制缓存
  channel_features_{train,test}.npz         逐轴箱特征张量 (N, 8, 8, 2, K)、车速、文件名
                                            （训练集另含标签）；供 train.py 使用
  features_{train,test}.pkl                 不减基线的聚合特征表 {"file": ..., "side": ...}；
                                            供 eda.py 使用

用法：
  python code/build_cache.py                 生成全部缓存
  python code/build_cache.py --no-raw-cache  不写 .npy
已存在的 .npy 会被直接复用；修改 features.py 后重新运行即可更新特征。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C
import data_io
import features

META_COLS = ("filename", "side", "label", "label_id", "side_fault")


def _replace_atomic(path: Path, writer) -> None:
    """先写临时文件再重命名，避免中断时留下不完整的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, path)


def process_file(task: tuple[str, bool]):
    """处理单个文件：读取（优先缓存）→ 必要时写缓存 → 计算逐轴箱特征。"""
    csv_path, write_raw = Path(task[0]), task[1]
    npy_path = data_io.raw_cache_path(csv_path)
    wrote = False
    if npy_path.exists():
        raw = np.load(npy_path, mmap_mode="r")
    else:
        raw = data_io.read_csv_array(csv_path)
        if write_raw:
            arr = np.ascontiguousarray(raw, dtype=np.float32)
            _replace_atomic(npy_path, lambda fh: np.save(fh, arr))
            wrote = True
    pulse, signals = data_io.split_speed_and_signals(raw)
    speed, cf = features.extract_channel_features(pulse, signals)
    return csv_path.name, speed, cf, wrote


def extract_split(files: list[Path], write_raw: bool, workers: int):
    tasks = [(str(p), write_raw) for p in files]
    names, speeds, cfs, n_written = [], [], [], 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (name, v, cf, wrote) in enumerate(ex.map(process_file, tasks, chunksize=4), 1):
            names.append(name)
            speeds.append(v)
            cfs.append(cf)
            n_written += int(wrote)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i:>3}/{len(tasks)}  {time.perf_counter() - t0:6.1f} s", flush=True)
    return names, np.asarray(speeds, dtype=np.float64), np.stack(cfs), n_written


def save_channel_npz(path: Path, names, speeds, cf, labels: pd.DataFrame | None) -> None:
    arrays = {
        "filename": np.array(names),
        "speed_mps": speeds,
        "cf": cf.astype(np.float32),
        "feature_names": np.array(features.CHANNEL_FEATURE_NAMES),
    }
    if labels is not None:
        lab = labels.set_index(C.LABELS_FILE_COL)
        arrays["label"] = lab.loc[names, C.LABELS_LABEL_COL].to_numpy().astype(str)
        arrays["label_id"] = lab.loc[names, "label_id"].to_numpy().astype(np.int64)
    _replace_atomic(path, lambda fh: np.savez(fh, **arrays))


def build_tables(names, speeds, cf, labels: pd.DataFrame | None) -> dict:
    """不减基线的聚合特征表，供 eda.py 使用。"""
    fx, fn = features.file_level(speeds, cf)
    sx, sn, fidx, sides = features.side_level(speeds, cf)

    df_file = pd.DataFrame(fx, columns=fn)
    df_file.insert(0, "filename", names)

    df_side = pd.DataFrame(sx, columns=sn)
    df_side.insert(0, "side", sides)
    df_side.insert(0, "filename", np.asarray(names)[fidx])

    if labels is not None:
        lab = labels.set_index(C.LABELS_FILE_COL)
        df_file.insert(1, "label", lab.loc[names, C.LABELS_LABEL_COL].to_numpy())
        df_file.insert(2, "label_id", lab.loc[names, "label_id"].to_numpy())
        file_label = df_side["filename"].map(lab[C.LABELS_LABEL_COL])
        df_side.insert(2, "label", file_label.to_numpy())
        df_side.insert(3, "side_fault", (file_label == df_side["side"]).astype(int).to_numpy())

    return {"file": df_file, "side": df_side}


def summarize(cf: np.ndarray, tables: dict, n_written: int, elapsed: float) -> None:
    f, s = tables["file"], tables["side"]
    fcols = [c for c in f.columns if c not in META_COLS]
    scols = [c for c in s.columns if c not in META_COLS]
    x = f[fcols].to_numpy(dtype=np.float64)
    sp = f["speed_mps"]
    print(f"  channel tensor : {cf.shape}, NaN = {int(np.isnan(cf).sum())}")
    print(f"  file table     : {f.shape[0]} rows x {len(fcols)} features")
    print(f"  side table     : {s.shape[0]} rows x {len(scols)} features")
    print(f"  NaN / inf      : {int(np.isnan(x).sum())} / {int(np.isinf(x).sum())}")
    print(f"  speed m/s      : min {sp.min():.3f}  median {sp.median():.3f}  max {sp.max():.3f}")
    print(f"  speed < {C.LOW_SPEED_RULE_MPS} m/s : {int((sp < C.LOW_SPEED_RULE_MPS).sum())} files")
    print(f"  npy written    : {n_written}")
    print(f"  elapsed        : {elapsed:.1f} s")
    if "label" in f.columns:
        print("  file labels    :", f["label"].value_counts().to_dict())
        print("  side_fault     :", s["side_fault"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description="Rail Corrugation 数据预处理与缓存")
    ap.add_argument("--no-raw-cache", action="store_true", help="不写 .npy")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="并行进程数")
    args = ap.parse_args()

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labels = data_io.load_labels()
    train_files = data_io.list_csv_files(C.TRAIN_DIR)
    test_files = data_io.list_csv_files(C.TEST_DIR)
    assert [p.name for p in train_files] == labels[C.LABELS_FILE_COL].tolist()

    for name, files, lab, npz_path, pkl_path in [
        ("train", train_files, labels, C.CHANNEL_FEATURES_TRAIN, C.FEATURES_TRAIN),
        ("test", test_files, None, C.CHANNEL_FEATURES_TEST, C.FEATURES_TEST),
    ]:
        print(f"[{name}] {len(files)} files, workers = {args.workers}")
        t0 = time.perf_counter()
        names, speeds, cf, n_written = extract_split(files, not args.no_raw_cache, args.workers)
        save_channel_npz(npz_path, names, speeds, cf, lab)
        tables = build_tables(names, speeds, cf, lab)
        pd.to_pickle(tables, pkl_path)
        summarize(cf, tables, n_written, time.perf_counter() - t0)
        print(f"  saved -> {npz_path.relative_to(C.WORKSPACE_ROOT)}")
        print(f"  saved -> {pkl_path.relative_to(C.WORKSPACE_ROOT)}")


if __name__ == "__main__":
    main()
