"""程序：推断。

输入：--input   单个 CSV 文件，或包含 CSV 文件的目录（格式与 Train/Test 相同）
      --model   模型文件（默认 model/rail_model.joblib）
输出：--output  预测结果 CSV（file_id, prediction），格式与 04_Example_Submission/rail_predictions.csv 相同；
                若给出的是目录，则写入该目录下的 rail_predictions.csv
      --details 可选：另存包含车速、低速规则标记、两侧故障分数的明细 CSV
流程：读取 → 逐轴箱特征 → 车速低于阈值的文件判为 Normal → 其余文件按训练时的传感器基线
      聚合特征，各组成模型给出两侧故障分数并取平均 → 按阈值判定。
本文件不依赖 train.py；其他程序（如应用界面）可直接调用 predict_files()。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

import config as C
import data_io
import features


# ============================================================ 判定（与训练时的判定规则相同）
def fault_scores(structure: str, proba: np.ndarray) -> np.ndarray:
    """统一为每个文件的两侧故障分数 (n, 2)。"""
    if structure == "file":
        return proba[:, 1:3]
    return proba[:, 1].reshape(-1, 2)


def decide(s: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """某侧分数不低于该侧阈值即判为该侧；两侧均满足时取超出比例较大者；否则 Normal。"""
    r1, r2 = s[:, 0] / t1, s[:, 1] / t2
    e1, e2 = r1 >= 1, r2 >= 1
    pred = np.zeros(len(s), dtype=int)
    pred[e1 & (~e2 | (r1 >= r2))] = 1
    pred[e2 & (~e1 | (r2 > r1))] = 2
    return pred


# ============================================================ 模型
def check_compat(bundle: dict) -> None:
    """模型文件须为组合格式，且记录的特征设定须与当前 config.py / features.py 一致。"""
    if "components" not in bundle or "thresholds" not in bundle:
        raise RuntimeError("模型文件为旧格式（单模型），与本推断程序不兼容，请用当前 train.py 重新训练")
    if tuple(bundle["channel_feature_names"]) != features.CHANNEL_FEATURE_NAMES:
        raise RuntimeError("模型的基础特征定义与当前 features.py 不一致，请重新训练模型")
    for key, value in bundle["settings"].items():
        if getattr(C, key) != value:
            raise RuntimeError(f"设定 {key} 与训练时不一致：当前 {getattr(C, key)}，训练时 {value}")


def load_bundle(path: Path = C.MODEL_FILE) -> dict:
    bundle = joblib.load(path)
    check_compat(bundle)
    return bundle


def _component_models(comp: dict):
    """兼容两种保存格式：{"ensemble": {"cols", "models"}} 或 {"model": 单个模型}。"""
    if "ensemble" in comp:
        return comp["ensemble"]["cols"], comp["ensemble"]["models"]
    return None, [comp["model"]]


# ============================================================ 输入
def resolve_inputs(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        files = data_io.list_csv_files(path)
    elif path.is_file() and path.suffix.lower() == ".csv":
        files = [path]
    else:
        raise FileNotFoundError(f"输入须为 CSV 文件或包含 CSV 的目录：{path}")
    if not files:
        raise FileNotFoundError(f"目录中没有 CSV 文件：{path}")
    return files


def extract(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """逐文件读取并计算逐轴箱特征；任何文件出错都会汇总报告。"""
    names, speeds, cfs, errors = [], [], [], []
    for p in paths:
        try:
            raw = data_io.read_csv_array(p)
            if raw.shape[0] < 2:
                raise ValueError(f"采样点过少：{raw.shape[0]} 行")
            pulse, signals = data_io.split_speed_and_signals(raw)
            v, cf = features.extract_channel_features(pulse, signals)
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"{Path(p).name}: {exc}")
            continue
        names.append(Path(p).name)
        speeds.append(v)
        cfs.append(cf)
    if errors:
        raise ValueError("以下文件无法处理：\n  " + "\n  ".join(errors))
    return names, np.asarray(speeds, dtype=np.float64), np.stack(cfs)


# ============================================================ 预测
def _aggregate(structure: str, speeds, cf, bundle: dict):
    cs = bundle.get("aggregation_car_features", "full")
    if structure == "file":
        return features.file_level(speeds, cf, bundle["baseline"], car_features=cs)
    x, names, _, _ = features.side_level(speeds, cf, bundle["baseline"], car_features=cs)
    return x, names


def moving_scores(bundle: dict, speeds: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """各组成模型的两侧故障分数取平均，返回 (n, 2)。"""
    agg_cache, all_scores = {}, []
    for comp in bundle["components"]:
        st = comp["structure"]
        if st not in agg_cache:
            agg_cache[st] = _aggregate(st, speeds, cf, bundle)
        x, names = agg_cache[st]
        if list(names) != list(comp["all_feature_names"]):
            raise RuntimeError(f"组成模型 {comp['name']} 的特征名与当前聚合结果不一致")
        xk = x[:, comp["keep_idx"]]
        cols, models = _component_models(comp)
        if cols is not None:
            xk = xk[:, cols]
        proba = np.mean([m.predict_proba(xk) for m in models], axis=0)
        all_scores.append(fault_scores(st, proba))
    return np.mean(all_scores, axis=0)


def predict_files(paths: list[Path], bundle: dict | None = None) -> pd.DataFrame:
    """对一组 CSV 文件做预测，返回 file_id、prediction 及明细列。"""
    bundle = bundle if bundle is not None else load_bundle()
    names, speeds, cf = extract(paths)
    low = speeds < bundle["low_speed_rule_mps"]
    pred = np.zeros(len(names), dtype=int)
    scores = np.full((len(names), 2), np.nan)
    moving = ~low
    if moving.any():
        s = moving_scores(bundle, speeds[moving], cf[moving])
        scores[moving] = s
        pred[moving] = decide(s, *bundle["thresholds"])
    class_names = bundle["class_names"]
    return pd.DataFrame({
        C.SUBMISSION_FILE_COL: names,
        C.SUBMISSION_PRED_COL: [class_names[k] for k in pred],
        "speed_mps": np.round(speeds, 3),
        "low_speed_rule": low,
        "p_side1_fault": scores[:, 0],
        "p_side2_fault": scores[:, 1],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Rail Corrugation 推断")
    ap.add_argument("--input", required=True, type=Path, help="CSV 文件或包含 CSV 的目录")
    ap.add_argument("--output", required=True, type=Path, help="输出 CSV 路径，或输出目录")
    ap.add_argument("--model", type=Path, default=C.MODEL_FILE, help="模型文件")
    ap.add_argument("--details", type=Path, default=None, help="可选：明细 CSV 路径")
    args = ap.parse_args()

    t0 = time.perf_counter()
    paths = resolve_inputs(args.input)
    bundle = load_bundle(args.model)
    df = predict_files(paths, bundle)

    out = args.output if args.output.suffix.lower() == ".csv" else args.output / C.SUBMISSION_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    df[[C.SUBMISSION_FILE_COL, C.SUBMISSION_PRED_COL]].to_csv(out, index=False)
    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.details, index=False)

    counts = df[C.SUBMISSION_PRED_COL].value_counts().reindex(bundle["class_names"], fill_value=0)
    print(f"files      : {len(df)}  (low-speed rule: {int(df['low_speed_rule'].sum())})")
    print(f"predicted  : {counts.to_dict()}")
    print(f"model      : source={bundle['source']}  thresholds={bundle['thresholds']}")
    print(f"output     : {out}")
    if args.details is not None:
        print(f"details    : {args.details}")
    print(f"elapsed    : {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
