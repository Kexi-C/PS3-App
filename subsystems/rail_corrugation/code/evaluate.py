"""程序：模型评测。

输入：CACHE_DIR/oof_predictions.csv（train.py 生成的折外预测）。
输出：终端打印的评测结果；错误明细与混淆矩阵图保存至 CACHE_DIR/evaluation/。
不训练模型，不修改任何已有文件。

用法：
  python code/evaluate.py                          评测最终判定（pred 列）
  python code/evaluate.py --pred-col pred_default  评测默认阈值下的判定
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config as C

EVAL_DIR = C.CACHE_DIR / "evaluation"
FAST_SPEED_MPS = 9.0
N_BOOT = 2000
LABELS = list(range(len(C.CLASS_NAMES)))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def macro_f1(y, p) -> float:
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def load(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"filename", "label", "speed_mps", "low_speed", pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列: {sorted(missing)}")
    for col in ("label", pred_col):
        unknown = set(df[col]) - set(C.CLASS_NAMES)
        if unknown:
            raise ValueError(f"列 {col} 中存在未知类别: {unknown}")
    df["low_speed"] = df["low_speed"].astype(str).str.lower().isin(("true", "1"))
    return df


def stratified_bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT,
                            seed: int = C.RANDOM_SEED) -> tuple[float, float]:
    """按真实类别分层重抽样，估计 macro F1 的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in LABELS]
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        scores[b] = macro_f1(y[idx], p[idx])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ============================================================ 1. 总体成绩
def overall(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> None:
    section("1. Overall")
    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "moving": ~df["low_speed"].to_numpy(),
        f"speed >= {FAST_SPEED_MPS:g}": (df["speed_mps"] >= FAST_SPEED_MPS).to_numpy(),
    }
    rows = []
    for name, m in subsets.items():
        per = f1_score(y[m], p[m], labels=LABELS, average=None, zero_division=0)
        rows.append({"subset": name, "n": int(m.sum()),
                     "macro_f1": macro_f1(y[m], p[m]),
                     "accuracy": accuracy_score(y[m], p[m]),
                     **{f"f1_{c}": v for c, v in zip(C.CLASS_NAMES, per)}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    lo, hi = stratified_bootstrap_ci(y, p)
    print(f"\nmacro F1 (all) 95% CI, stratified bootstrap x{N_BOOT}: [{lo:.3f}, {hi:.3f}]")

    low = df["low_speed"].to_numpy()
    print(f"low-speed rule: {int(low.sum())} files, "
          f"labels {df.loc[low, 'label'].value_counts().to_dict()}, "
          f"predicted {pd.Series(p[low]).map(dict(enumerate(C.CLASS_NAMES))).value_counts().to_dict()}")


# ============================================================ 2. 各类别指标
def per_class(y: np.ndarray, p: np.ndarray) -> None:
    section("2. Per-class metrics (all files)")
    print(classification_report(y, p, labels=LABELS, target_names=list(C.CLASS_NAMES),
                                digits=3, zero_division=0))


# ============================================================ 3. 混淆矩阵
def confusion(y: np.ndarray, p: np.ndarray, moving: np.ndarray) -> np.ndarray:
    section("3. Confusion matrix (rows = true, columns = predicted)")
    for name, m in (("all", np.ones(len(y), dtype=bool)), ("moving", moving)):
        cm = confusion_matrix(y[m], p[m], labels=LABELS)
        tab = pd.DataFrame(cm, index=[f"true {c}" for c in C.CLASS_NAMES],
                           columns=[f"pred {c}" for c in C.CLASS_NAMES])
        print(f"\n[{name}]")
        print(tab.to_string())
    return confusion_matrix(y, p, labels=LABELS)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(LABELS, C.CLASS_NAMES)
    ax.set_yticks(LABELS, C.CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================ 4. 错误分析
def error_analysis(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, pred_col: str) -> None:
    section("4. Error analysis")
    kind = np.full(len(y), "", dtype=object)
    kind[(y > 0) & (p == 0)] = "missed fault"
    kind[(y == 0) & (p > 0)] = "false alarm"
    kind[(y > 0) & (p > 0) & (y != p)] = "wrong side"
    err = kind != ""
    print(pd.Series(kind[err]).value_counts().reindex(
        ["missed fault", "false alarm", "wrong side"], fill_value=0).to_string())

    prob_cols = [c for c in df.columns if c.startswith("p_")]
    cols = ["filename", "label", pred_col, "speed_mps"] + prob_cols
    table = df.loc[err, cols].copy()
    table.insert(0, "error", kind[err])
    table = table.sort_values(["error", "label", "speed_mps"])
    print()
    print(table.round(3).to_string(index=False) if len(table) else "(no errors)")
    table.to_csv(EVAL_DIR / "errors.csv", index=False)

    print("\nspeed (m/s) of moving files, median [min, max]:")
    moving = ~df["low_speed"].to_numpy()
    sp = df["speed_mps"].to_numpy()
    for k, cls in enumerate(C.CLASS_NAMES):
        for tag, m in (("correct", (y == k) & (p == k) & moving),
                       ("error", (y == k) & (p != k) & moving)):
            if m.any():
                print(f"  {cls:<8} {tag:<8} n={int(m.sum()):>3}  "
                      f"{np.median(sp[m]):6.2f} [{sp[m].min():5.2f}, {sp[m].max():5.2f}]")


# ============================================================ 5. 两种判定对比
def compare_decisions(df: pd.DataFrame, y: np.ndarray) -> None:
    cols = [c for c in ("pred_default", "pred") if c in df.columns]
    if len(cols) < 2:
        return
    section("5. Default vs tuned decision (all files)")
    for c in cols:
        pc = df[c].map(C.LABEL_TO_ID).to_numpy()
        per = f1_score(y, pc, labels=LABELS, average=None, zero_division=0)
        print(f"  {c:<13} macro F1 = {macro_f1(y, pc):.3f}   per-class F1 = {np.round(per, 3).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rail Corrugation 模型评测")
    ap.add_argument("--oof", type=Path, default=C.OOF_PREDICTIONS, help="折外预测文件")
    ap.add_argument("--pred-col", default="pred", help="要评测的预测列（pred 或 pred_default）")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load(args.oof, args.pred_col)
    y = df["label"].map(C.LABEL_TO_ID).to_numpy()
    p = df[args.pred_col].map(C.LABEL_TO_ID).to_numpy()
    print(f"source: {args.oof.name}   column: {args.pred_col}   files: {len(df)}")

    overall(df, y, p)
    per_class(y, p)
    cm = confusion(y, p, ~df["low_speed"].to_numpy())
    plot_confusion(cm, f"OOF confusion matrix ({args.pred_col})", EVAL_DIR / "confusion_matrix.png")
    error_analysis(df, y, p, args.pred_col)
    compare_decisions(df, y)

    section("Saved files")
    for f in sorted(EVAL_DIR.iterdir()):
        print(" ", f.relative_to(C.WORKSPACE_ROOT))


if __name__ == "__main__":
    main()
