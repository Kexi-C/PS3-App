"""程序：模型训练。

输入：CACHE_DIR/channel_features_train.npz（逐轴箱特征、车速、标签）。
输出：
  CACHE_DIR/cv_results.csv        各方案的交叉验证成绩
  CACHE_DIR/oof_predictions.csv   选定方案的折外分数与判定（供 evaluate.py 使用）
  model/rail_model.joblib         以全部运动文件重新训练的最终模型
流程：
  1. 车速低于 LOW_SPEED_RULE_MPS 的文件由规则判为 Normal，不参与训练；
  2. 重复分层 K 折交叉验证（随机种子固定，各次运行的划分完全一致）；每一折内部估计
     传感器基线、聚合特征，训练三个基础模型：
       side_inv    单侧视图 + 位置无关车厢特征
       side_inv_w  同上，Side I 与 Side II 故障样本加权至总分量相等
       file_full   文件级三分类 + 全部车厢特征
  3. 每个模型给出每个文件的两侧故障分数 (s1, s2)；另构造两个组合（单侧视图模型与
     file_full 取平均），共 5 种分数来源；
  4. 每种来源比较两种判定方式：两侧共用阈值 / 两侧各自阈值，共 10 个方案；
     成绩以嵌套方式估计：每一折的阈值只用其余各折的分数调整；
  5. 选择嵌套成绩最高的方案（相同时优先共用阈值、单个模型），以全部运动文件
     重新训练所需模型，并在全部折外分数上确定最终阈值。
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

import config as C
import features

# 基础模型：名称 -> (结构, 车厢特征组合, 是否对两侧故障样本加权)
BASE_MODELS = {
    "side_inv": ("side", "invariant", False),
    "side_inv_w": ("side", "invariant", True),
    "file_full": ("file", "full", False),
}
# 分数来源：名称 -> 参与平均的基础模型
SOURCES = {
    "side_inv": ("side_inv",),
    "side_inv_w": ("side_inv_w",),
    "file_full": ("file_full",),
    "ens(side_inv+file_full)": ("side_inv", "file_full"),
    "ens(side_inv_w+file_full)": ("side_inv_w", "file_full"),
}
THRESHOLD_MODES = ("shared", "per_side")
T_GRID = np.round(np.arange(0.02, 0.981, 0.02), 2)
FAST_SPEED_MPS = 9.0
CV_RESULTS = C.CACHE_DIR / "cv_results.csv"
N_CLASSES = len(C.CLASS_NAMES)
_CAR_SPECIFIC = re.compile(r"^c\d+_")

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================ 数据
def load_train():
    d = np.load(C.CHANNEL_FEATURES_TRAIN)
    assert tuple(d["feature_names"]) == features.CHANNEL_FEATURE_NAMES, \
        "缓存中的特征定义与 features.py 不一致，请重新运行 build_cache.py"
    return (d["filename"].astype(str), d["speed_mps"].astype(np.float64),
            d["cf"], d["label"].astype(str), d["label_id"].astype(int))


def aggregate(structure: str, speeds, cf, baseline):
    """聚合完整特征（含全部车厢级特征）；各特征组合由 keep_columns 截取。"""
    if structure == "file":
        return features.file_level(speeds, cf, baseline, car_features="full")
    x, names, _, _ = features.side_level(speeds, cf, baseline, car_features="full")
    return x, names


def keep_columns(names: list[str], car_features: str) -> np.ndarray:
    """车速本身不作为特征；按车厢级特征组合截取列。"""
    keep = []
    for i, n in enumerate(names):
        if n == "speed_mps":
            continue
        if car_features == "none" and (n.startswith("k_") or _CAR_SPECIFIC.match(n)):
            continue
        if car_features == "invariant" and _CAR_SPECIFIC.match(n):
            continue
        keep.append(i)
    return np.array(keep)


def side_targets(y: np.ndarray) -> np.ndarray:
    """文件标签 -> 单侧视图标签（行顺序：文件0 Side I, 文件0 Side II, 文件1 Side I, ...）。"""
    return np.column_stack([y == 1, y == 2]).reshape(-1).astype(int)


def row_targets(structure: str, y: np.ndarray) -> np.ndarray:
    return y if structure == "file" else side_targets(y)


def side_weights(y_rows: np.ndarray) -> np.ndarray:
    """单侧视图样本权重：使 Side I 与 Side II 故障样本的总权重相等，正常样本权重为 1。"""
    side = np.tile([1, 2], len(y_rows) // 2)
    pos = y_rows == 1
    n1, n2 = int((pos & (side == 1)).sum()), int((pos & (side == 2)).sum())
    w = np.ones(len(y_rows), dtype=np.float64)
    if n1 and n2:
        w[pos & (side == 1)] = 0.5 * (n1 + n2) / n1
        w[pos & (side == 2)] = 0.5 * (n1 + n2) / n2
    return w


# ============================================================ 模型
def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=5,
            l2_regularization=1.0, class_weight="balanced", random_state=C.RANDOM_SEED),
    )


def fit_model(structure: str, weighted: bool, x: np.ndarray, y_rows: np.ndarray):
    model = make_model()
    if weighted:
        model.fit(x, y_rows, histgradientboostingclassifier__sample_weight=side_weights(y_rows))
    else:
        model.fit(x, y_rows)
    return model


def fault_scores(structure: str, proba: np.ndarray) -> np.ndarray:
    """统一为每个文件的两侧故障分数 (n, 2)。"""
    if structure == "file":
        return proba[:, 1:3]
    return proba[:, 1].reshape(-1, 2)


# ============================================================ 判定与评分
def decide(s: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """某侧分数不低于该侧阈值即判为该侧；两侧均满足时取超出比例较大者；否则 Normal。"""
    r1, r2 = s[:, 0] / t1, s[:, 1] / t2
    e1, e2 = r1 >= 1, r2 >= 1
    pred = np.zeros(len(s), dtype=int)
    pred[e1 & (~e2 | (r1 >= r2))] = 1
    pred[e2 & (~e1 | (r2 > r1))] = 2
    return pred


def per_class_f1(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    cm = np.bincount(N_CLASSES * y + p, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(cm).astype(np.float64)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(per_class_f1(y, p).mean())


def tune(s: np.ndarray, y: np.ndarray, mode: str) -> tuple[float, float]:
    if mode == "shared":
        cands = [(t, t) for t in T_GRID]
    else:
        cands = [(a, b) for a in T_GRID for b in T_GRID]
    return max(cands, key=lambda t: (macro_f1(y, decide(s, *t)),
                                     -(abs(t[0] - 0.5) + abs(t[1] - 0.5))))


def embed(pred_m: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """运动文件的判定嵌回全部文件；低速文件按规则判为 Normal (0)。"""
    out = np.zeros(len(moving), dtype=int)
    out[moving] = pred_m
    return out


def nested(s, y, y_all, moving, fold_ids, mode):
    """嵌套估计：每一折的阈值只用同一次重复中其余各折的分数调整。"""
    f1s, per, preds = [], [], []
    for r in range(fold_ids.shape[0]):
        pred = np.zeros(len(y), dtype=int)
        for f in np.unique(fold_ids[r]):
            va = fold_ids[r] == f
            t = tune(s[~va], y[~va], mode)
            pred[va] = decide(s[va], *t)
        pa = embed(pred, moving)
        f1s.append(macro_f1(y_all, pa))
        per.append(per_class_f1(y_all, pa))
        preds.append(pa)
    return np.mean(f1s), np.std(f1s), np.mean(per, axis=0), preds[0]


# ============================================================ 交叉验证
def run_cv(speeds, cf, y):
    n = len(y)
    scores = {m: np.full((C.CV_N_REPEATS, n, 2), np.nan) for m in BASE_MODELS}
    fold_ids = np.full((C.CV_N_REPEATS, n), -1)
    rskf = RepeatedStratifiedKFold(n_splits=C.CV_N_SPLITS, n_repeats=C.CV_N_REPEATS,
                                   random_state=C.RANDOM_SEED)
    n_splits = C.CV_N_SPLITS * C.CV_N_REPEATS
    structures = tuple(dict.fromkeys(st for st, _, _ in BASE_MODELS.values()))
    t0 = time.perf_counter()
    for i, (tr, va) in enumerate(rskf.split(np.zeros(n), y)):
        rep, fold = divmod(i, C.CV_N_SPLITS)
        fold_ids[rep, va] = fold
        base = features.fit_baseline(cf[tr])
        mats = {}
        for st in structures:
            xtr, names = aggregate(st, speeds[tr], cf[tr], base)
            xva, _ = aggregate(st, speeds[va], cf[va], base)
            mats[st] = (xtr, xva, names)
        for mname, (st, cs, weighted) in BASE_MODELS.items():
            xtr, xva, names = mats[st]
            keep = keep_columns(names, cs)
            model = fit_model(st, weighted, xtr[:, keep], row_targets(st, y[tr]))
            scores[mname][rep, va] = fault_scores(st, model.predict_proba(xva[:, keep]))
        print(f"  fold {i + 1:>2}/{n_splits}  elapsed {time.perf_counter() - t0:6.1f} s (cumulative)",
              flush=True)
    return scores, fold_ids


# ============================================================ 主流程
def main() -> None:
    t_start = time.perf_counter()
    fnames, speeds, cf, labels, y_all = load_train()
    moving = speeds >= C.LOW_SPEED_RULE_MPS
    y, speeds_m, cf_m = y_all[moving], speeds[moving], cf[moving]
    print(f"files: {len(y_all)}  moving: {int(moving.sum())}  "
          f"low-speed (rule -> Normal): {int((~moving).sum())}")
    print("moving labels:", pd.Series(labels[moving]).value_counts().to_dict())

    print(f"\n[cross-validation] {len(BASE_MODELS)} base models, "
          f"{C.CV_N_REPEATS} x {C.CV_N_SPLITS}-fold")
    scores, fold_ids = run_cv(speeds_m, cf_m, y)
    avg = {m: s.mean(axis=0) for m, s in scores.items()}          # 各基础模型：重复平均后的分数

    rows, source_scores = [], {}
    for src, comps in SOURCES.items():
        s = np.mean([avg[m] for m in comps], axis=0)
        source_scores[src] = s
        for mode in THRESHOLD_MODES:
            t = tune(s, y, mode)
            f1_nest, f1_nest_std, per_nest, _ = nested(s, y, y_all, moving, fold_ids, mode)
            rows.append({
                "source": src, "thresholds": mode, "n_models": len(comps),
                "f1_default": macro_f1(y_all, embed(decide(s, 0.5, 0.5), moving)),
                "t_side1": t[0], "t_side2": t[1],
                "f1_tuned": macro_f1(y_all, embed(decide(s, *t), moving)),
                "f1_nested": f1_nest, "f1_nested_std": f1_nest_std,
                "nested_f1_SideI": per_nest[1], "nested_f1_SideII": per_nest[2],
            })
    results = pd.DataFrame(rows).sort_values(
        ["f1_nested", "thresholds", "n_models"], ascending=[False, True, True]).reset_index(drop=True)
    results.to_csv(CV_RESULTS, index=False)
    print("\n[cv results] macro F1 on all files "
          "(tuned = thresholds tuned on all OOF, optimistic; nested = thresholds tuned on other folds)")
    print(results.round(3).to_string(index=False))

    sel = results.iloc[0]
    src, mode = sel["source"], sel["thresholds"]
    comps = SOURCES[src]
    t = (float(sel["t_side1"]), float(sel["t_side2"]))
    print(f"\n[selected] source={src}  thresholds={mode}  t=({t[0]:.2f}, {t[1]:.2f})  "
          f"f1_nested={sel['f1_nested']:.3f}  f1_tuned={sel['f1_tuned']:.3f}")

    # ---- 选定方案的折外分数与判定
    s = source_scores[src]
    _, _, _, pred_nest = nested(s, y, y_all, moving, fold_ids, mode)
    oof_df = pd.DataFrame({"filename": fnames, "label": labels, "speed_mps": speeds,
                           "low_speed": ~moving})
    for j, col in enumerate(["p_side1_fault", "p_side2_fault"]):
        oof_df[col] = np.nan
        oof_df.loc[moving, col] = s[:, j]
    oof_df["pred_default"] = [C.CLASS_NAMES[k] for k in embed(decide(s, 0.5, 0.5), moving)]
    oof_df["pred"] = [C.CLASS_NAMES[k] for k in embed(decide(s, *t), moving)]
    oof_df["pred_nested"] = [C.CLASS_NAMES[k] for k in pred_nest]
    oof_df.to_csv(C.OOF_PREDICTIONS, index=False)

    # ---- 最终模型：全部运动文件
    t0 = time.perf_counter()
    base = features.fit_baseline(cf_m)
    agg_cache, components = {}, []
    for mname in comps:
        st, cs, weighted = BASE_MODELS[mname]
        if st not in agg_cache:
            agg_cache[st] = aggregate(st, speeds_m, cf_m, base)
        x, names = agg_cache[st]
        keep = keep_columns(names, cs)
        model = fit_model(st, weighted, x[:, keep], row_targets(st, y))
        components.append({
            "name": mname, "structure": st, "car_features": cs, "weighted": weighted,
            "model": model, "all_feature_names": names, "keep_idx": keep,
            "feature_names": [names[i] for i in keep],
        })
    bundle = {
        "source": src, "threshold_mode": mode, "thresholds": t,
        "components": components,
        "aggregation_car_features": "full",   # 推断时先聚合完整特征，再按各模型的 keep_idx 截取
        "baseline": base.astype(np.float32),
        "low_speed_rule_mps": C.LOW_SPEED_RULE_MPS,
        "class_names": C.CLASS_NAMES,
        "channel_feature_names": features.CHANNEL_FEATURE_NAMES,
        "settings": {"AGG_STATS": C.AGG_STATS, "CONTRAST_STATS": C.CONTRAST_STATS,
                     "PAIR_STATS": C.PAIR_STATS, "CAR_STATS": C.CAR_STATS,
                     "AXLE_PAIRS": C.AXLE_PAIRS, "FREQ_BANDS_HZ": C.FREQ_BANDS_HZ,
                     "WAVELENGTH_BANDS_M": C.WAVELENGTH_BANDS_M,
                     "WELCH_NPERSEG": C.WELCH_NPERSEG},
        "cv": sel.to_dict(),
    }
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, C.MODEL_FILE, compress=3)

    print(f"\n[final model] {len(components)} model(s) trained on {int(moving.sum())} moving files, "
          f"{time.perf_counter() - t0:.1f} s")
    print(f"  model size : {C.MODEL_FILE.stat().st_size / 1e6:.2f} MB")
    for p in (CV_RESULTS, C.OOF_PREDICTIONS, C.MODEL_FILE):
        print(f"  saved -> {p.relative_to(C.WORKSPACE_ROOT)}")
    print(f"\nelapsed: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
