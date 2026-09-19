"""程序：探索性数据分析（EDA）。

输入：features_train.pkl、features_test.pkl，以及训练集的 .npy 原始信号缓存（只读）。
输出：终端打印的数值表格；图表与明细表保存至 CACHE_DIR/eda/（位于仓库之外）。
不修改任何已有缓存，不训练模型。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.metrics import roc_auc_score

import config as C
import data_io
import features

EDA_DIR = C.CACHE_DIR / "eda"
META_COLS = ("filename", "label", "label_id")
COLORS = {"Normal": "tab:gray", "Side I": "tab:blue", "Side II": "tab:red"}
LAMBDA_GRID_M = np.logspace(np.log10(0.01), np.log10(1.0), 200)
TOP_N = 15

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


# ============================================================ 1. 车速分布
def speed_analysis(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    section("1. Speed distribution")
    low = tr["speed_mps"] < C.MIN_SPEED_MPS
    tab = pd.crosstab(tr["label"], low.map({True: f"speed < {C.MIN_SPEED_MPS}",
                                             False: f"speed >= {C.MIN_SPEED_MPS}"}),
                      margins=True)
    print(tab.to_string())
    desc = tr.groupby("label")["speed_mps"].describe()[["count", "min", "25%", "50%", "75%", "max"]]
    print("\nspeed (m/s) by class:")
    print(desc.round(3).to_string())
    print(f"\nexactly 0 m/s (train): {int((tr['speed_mps'] == 0).sum())}")
    print(f"test speed < {C.MIN_SPEED_MPS}: {int((te['speed_mps'] < C.MIN_SPEED_MPS).sum())} / {len(te)}")

    tr.loc[low, ["filename", "label", "speed_mps"]].to_csv(EDA_DIR / "low_speed_files.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 21, 1.0)
    for cls in C.CLASS_NAMES:
        v = tr.loc[tr["label"] == cls, "speed_mps"]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                color=COLORS[cls], label=f"{cls} (n={len(v)})")
    ax.axvline(C.MIN_SPEED_MPS, color="k", ls="--", lw=0.8)
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("density")
    ax.set_title("Train speed distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "speed_by_class.png", dpi=130)
    plt.close(fig)


# ============================================================ 2. 缺失值核实
def nan_analysis(tr: pd.DataFrame) -> None:
    section("2. NaN check (files with speed >= threshold)")
    fcols = feature_cols(tr)
    n_nan = tr[fcols].isna().sum(axis=1)
    print(f"files with NaN: {int((n_nan > 0).sum())}  "
          f"(speed < {C.MIN_SPEED_MPS}: {int(((n_nan > 0) & (tr['speed_mps'] < C.MIN_SPEED_MPS)).sum())})")

    pat = re.compile(r"(wl\d+_\d+cm|dom_wavelength_m)")
    rows = []
    for i in tr.index[(n_nan > 0) & (tr["speed_mps"] >= C.MIN_SPEED_MPS)]:
        cols = [c for c in fcols if pd.isna(tr.at[i, c])]
        tags = sorted({m.group(1) for c in cols for m in [pat.search(c)] if m})
        rows.append({"filename": tr.at[i, "filename"], "label": tr.at[i, "label"],
                     "speed_mps": round(tr.at[i, "speed_mps"], 3),
                     "n_nan": int(n_nan[i]), "bands": " ".join(tags)})
    out = pd.DataFrame(rows, columns=["filename", "label", "speed_mps", "n_nan", "bands"])
    print(out.to_string(index=False) if len(out) else "(none)")
    print(f"total NaN in these files: {int(out['n_nan'].sum()) if len(out) else 0}")
    out.to_csv(EDA_DIR / "nan_files_moving.csv", index=False)


# ============================================================ 3. 单特征区分度
def separability(tr: pd.DataFrame) -> None:
    section(f"3. Single-feature AUC vs Normal (speed >= {C.MIN_SPEED_MPS})")
    d = tr[tr["speed_mps"] >= C.MIN_SPEED_MPS]
    print("files used:", d["label"].value_counts().to_dict())

    rows = []
    for c in feature_cols(d):
        rec = {"feature": c}
        for cls, key in (("Side I", "auc_side1"), ("Side II", "auc_side2")):
            sub = d.loc[d["label"].isin(["Normal", cls]), [c, "label"]].dropna()
            y = (sub["label"] == cls).astype(int)
            rec[key] = roc_auc_score(y, sub[c]) if y.nunique() == 2 else np.nan
        rows.append(rec)
    auc = pd.DataFrame(rows)
    auc["strength_side1"] = (auc["auc_side1"] - 0.5).abs()
    auc["strength_side2"] = (auc["auc_side2"] - 0.5).abs()
    auc.to_csv(EDA_DIR / "feature_auc.csv", index=False)

    for cls, a, s in (("Side I", "auc_side1", "strength_side1"),
                      ("Side II", "auc_side2", "strength_side2")):
        top = auc.nlargest(TOP_N, s)[["feature", "auc_side1", "auc_side2"]]
        print(f"\ntop {TOP_N} for {cls} vs Normal:")
        print(top.round(3).to_string(index=False))


# ============================================================ 4. 两侧功率谱差异
def compute_side_psd(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 f (F,), logp (N, 2 侧, 2 通道, F), speeds (N,)。每侧取 32 个轴箱功率谱的中位数。"""
    side_idx = (list(C.SIDE_I_POS_IDX), list(C.SIDE_II_POS_IDX))
    logp, speeds, f = [], [], None
    t0 = time.perf_counter()
    for i, p in enumerate(files, 1):
        pulse, sig = data_io.load_file(p)
        x = np.asarray(sig, dtype=np.float64)
        x = x - x.mean(axis=-1, keepdims=True)
        f, pxx = sps.welch(x, fs=C.FS_HZ, nperseg=C.WELCH_NPERSEG,
                           noverlap=C.WELCH_NOVERLAP, axis=-1)
        per_side = [np.median(pxx[:, idx].reshape(-1, C.N_CHANNEL_TYPES, len(f)), axis=0)
                    for idx in side_idx]
        logp.append(np.log10(np.stack(per_side) + features.EPS))
        speeds.append(features.compute_speed_mps(pulse))
        if i % 50 == 0 or i == len(files):
            print(f"  psd {i:>3}/{len(files)}  {time.perf_counter() - t0:5.1f} s", flush=True)
    return f, np.stack(logp), np.asarray(speeds)


def side_psd_analysis(tr: pd.DataFrame) -> None:
    section(f"4. Side I - Side II PSD difference in dB (speed >= {C.MIN_SPEED_MPS}, median over files)")
    d = tr[tr["speed_mps"] >= C.MIN_SPEED_MPS].reset_index(drop=True)
    files = [C.TRAIN_DIR / n for n in d["filename"]]
    f, logp, speeds = compute_side_psd(files)
    assert np.allclose(speeds, d["speed_mps"].to_numpy())

    diff_db = 10.0 * (logp[:, 0] - logp[:, 1])            # (N, 2 通道, F)
    labels = d["label"].to_numpy()

    # ---- 频率域频带表
    for ci, ch in enumerate(C.CHANNEL_TYPES):
        tab = {}
        for lo, hi in C.FREQ_BANDS_HZ:
            m = (f >= lo) & (f < hi)
            band = diff_db[:, ci, m].mean(axis=-1)
            tab[f"{lo}-{hi} Hz"] = {cls: np.median(band[labels == cls]) for cls in C.CLASS_NAMES}
        print(f"\n[{ch}] frequency bands")
        print(pd.DataFrame(tab).T.round(2).to_string())

    # ---- 波长域：按各文件车速把频率换算为波长后插值到统一波长网格
    lam_diff = np.full((len(d), C.N_CHANNEL_TYPES, len(LAMBDA_GRID_M)), np.nan)
    f_lo = C.FREQ_BANDS_HZ[0][0]
    for n, v in enumerate(speeds):
        fq = v / LAMBDA_GRID_M
        ok = (fq >= f_lo) & (fq <= f[-1])
        for ci in range(C.N_CHANNEL_TYPES):
            lam_diff[n, ci, ok] = np.interp(fq[ok], f, diff_db[n, ci])

    for ci, ch in enumerate(C.CHANNEL_TYPES):
        tab = {}
        for lo, hi in C.WAVELENGTH_BANDS_M:
            m = (LAMBDA_GRID_M >= lo) & (LAMBDA_GRID_M < hi)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    band = np.nanmean(lam_diff[:, ci, m], axis=-1)
                    tab[f"{lo * 100:g}-{hi * 100:g} cm"] = {
                        cls: np.nanmedian(band[labels == cls]) for cls in C.CLASS_NAMES}
        print(f"\n[{ch}] wavelength bands")
        print(pd.DataFrame(tab).T.round(2).to_string())

    # ---- 图：各类别两侧功率谱
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ci, ch in enumerate(C.CHANNEL_TYPES):
        for si, side in enumerate(C.SIDES):
            ax = axes[ci, si]
            for cls in C.CLASS_NAMES:
                ax.plot(f[1:], 10 * np.median(logp[labels == cls, si, ci, 1:], axis=0),
                        color=COLORS[cls], lw=1.2, label=cls)
            ax.set_xscale("log")
            ax.set_xlim(C.FREQ_BANDS_HZ[0][0], C.FS_HZ / 2)
            ax.set_title(f"{ch} / {side}")
            ax.set_ylabel("PSD (dB)")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "psd_by_class.png", dpi=130)
    plt.close(fig)

    # ---- 图：两侧差异（频率域 / 波长域）
    import warnings
    for name, x, y, xlabel, xlim in (
        ("psd_side_diff_freq.png", f[1:], diff_db[..., 1:], "frequency (Hz)",
         (C.FREQ_BANDS_HZ[0][0], C.FS_HZ / 2)),
        ("psd_side_diff_wavelength.png", LAMBDA_GRID_M * 100, lam_diff, "wavelength (cm)", (1, 100)),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ci, ch in enumerate(C.CHANNEL_TYPES):
            ax = axes[ci]
            for cls in C.CLASS_NAMES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ax.plot(x, np.nanmedian(y[labels == cls, ci], axis=0),
                            color=COLORS[cls], lw=1.2, label=cls)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{ch}: Side I - Side II")
        axes[0].set_ylabel("PSD difference (dB)")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(EDA_DIR / name, dpi=130)
        plt.close(fig)


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_pickle(C.FEATURES_TRAIN)["file"]
    te = pd.read_pickle(C.FEATURES_TEST)["file"]
    t0 = time.perf_counter()

    speed_analysis(tr, te)
    nan_analysis(tr)
    separability(tr)
    side_psd_analysis(tr)

    section("Saved files")
    for p in sorted(EDA_DIR.iterdir()):
        print(" ", p.relative_to(C.WORKSPACE_ROOT))
    print(f"\nelapsed: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
