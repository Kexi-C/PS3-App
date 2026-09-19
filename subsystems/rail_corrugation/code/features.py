"""特征提取与车速校正（纯函数，不涉及文件读写）。

分三部分：
1. 逐轴箱特征：extract_channel_features(pulse, signals) -> (车速, (8, 8, 2, K))
2. 聚合特征：file_level / side_level，输入一批逐轴箱特征 (N, 8, 8, 2, K)，
   可选传入传感器基线 baseline (8, 8, 2, K)，先减基线再跨轴箱聚合；
   car_features 控制车厢级特征："none" 不加，"invariant" 加位置无关特征（k_ 前缀），
   "full" 再加逐车厢特征（c1_ ~ c8_ 前缀）。
3. 车速残差化：fit_speed_residual / apply_speed_residual。
基线等由数据估计的参数须在交叉验证每一折内部拟合，避免信息泄漏。
"""
from __future__ import annotations

import re
import warnings

import numpy as np
from scipy import signal as sps
from scipy import stats

import config as C

EPS = 1e-12
CH_TAGS = ("vib", "shk")                         # 与 C.CHANNEL_TYPES 顺序对应
RESIDUAL_EXCLUDE = ("speed_mps", "is_side_II")   # 不做车速残差化的列
_SIDE_I = list(C.SIDE_I_POS_IDX)
_SIDE_II = list(C.SIDE_II_POS_IDX)
_PAIR_I = [a - 1 for a, _ in C.AXLE_PAIRS]
_PAIR_II = [b - 1 for _, b in C.AXLE_PAIRS]


def _channel_feature_names() -> tuple[str, ...]:
    names = ["log_rms", "log_p2p", "crest", "kurtosis", "skewness"]
    for lo, hi in C.FREQ_BANDS_HZ:
        names += [f"logE_f{lo}_{hi}", f"relE_f{lo}_{hi}"]
    for lo, hi in C.WAVELENGTH_BANDS_M:
        tag = f"wl{round(lo * 100)}_{round(hi * 100)}cm"
        names += [f"logE_{tag}", f"relE_{tag}"]
    names += ["dom_freq_hz", "dom_wavelength_m"]
    return tuple(names)


CHANNEL_FEATURE_NAMES = _channel_feature_names()
K = len(CHANNEL_FEATURE_NAMES)


# ============================================================ 第一部分：逐轴箱特征
def compute_speed_mps(pulse: np.ndarray) -> float:
    """由测速齿轮脉冲的 0/1 跳变次数计算车速（m/s）。"""
    n_transitions = int(np.count_nonzero(np.diff(np.asarray(pulse))))
    duration_s = len(pulse) / C.FS_HZ
    revs_per_s = n_transitions / C.TRANSITIONS_PER_REV / duration_s
    return float(revs_per_s * C.WHEEL_CIRCUMFERENCE_M)


def channel_features(signals: np.ndarray, speed_mps: float) -> np.ndarray:
    """对每一路信号计算特征，返回 (8, 8, 2, K)。"""
    x = np.asarray(signals, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)                 # 去除直流偏置
    t = x.shape[-1]

    # ---- 时域
    rms = np.sqrt(np.mean(x * x, axis=-1))
    valid = rms > EPS
    p2p = x.max(axis=-1) - x.min(axis=-1)
    with np.errstate(all="ignore"):
        crest = np.where(valid, np.abs(x).max(axis=-1) / (rms + EPS), 0.0)
        kurt = np.where(valid, stats.kurtosis(x, axis=-1, fisher=True, bias=False), 0.0)
        skew = np.where(valid, stats.skew(x, axis=-1, bias=False), 0.0)
    feats = [np.log10(rms + EPS), np.log10(p2p + EPS), crest, kurt, skew]

    # ---- 频域（Welch 功率谱密度）
    nperseg = min(C.WELCH_NPERSEG, t)
    noverlap = min(C.WELCH_NOVERLAP, nperseg // 2)
    f, pxx = sps.welch(x, fs=C.FS_HZ, nperseg=nperseg, noverlap=noverlap, axis=-1)
    df = f[1] - f[0]
    f_min = C.FREQ_BANDS_HZ[0][0]
    total = pxx[..., f >= f_min].sum(axis=-1) * df + EPS

    def band_energy(lo: float, hi: float) -> np.ndarray:
        mask = (f >= lo) & (f < hi)
        if not mask.any():
            return np.full(rms.shape, np.nan)
        return pxx[..., mask].sum(axis=-1) * df

    for lo, hi in C.FREQ_BANDS_HZ:
        e = band_energy(lo, hi)
        feats += [np.log10(e + EPS), e / total]

    # ---- 波长域：lambda = v / f
    speed_ok = speed_mps >= C.MIN_SPEED_MPS
    for lam_lo, lam_hi in C.WAVELENGTH_BANDS_M:
        if speed_ok:
            e = band_energy(speed_mps / lam_hi, speed_mps / lam_lo)
        else:
            e = np.full(rms.shape, np.nan)
        feats += [np.log10(e + EPS), e / total]

    # ---- 主频与主波长
    search = f >= f_min
    dom_f = f[search][np.argmax(pxx[..., search], axis=-1)]
    dom_wl = speed_mps / dom_f if speed_ok else np.full(rms.shape, np.nan)
    feats += [dom_f, dom_wl]

    return np.stack(feats, axis=-1)


def extract_channel_features(pulse: np.ndarray, signals: np.ndarray) -> tuple[float, np.ndarray]:
    """单个文件：返回 (车速 m/s, 逐轴箱特征 (8, 8, 2, K) float32)。"""
    v = compute_speed_mps(pulse)
    return v, channel_features(signals, v).astype(np.float32)


# ============================================================ 第二部分：基线与聚合
def fit_baseline(cf: np.ndarray) -> np.ndarray:
    """各传感器、各特征在一批文件上的中位数，返回 (8, 8, 2, K)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.asarray(cf, dtype=np.float64), axis=0)


def _reduce(x: np.ndarray, stat: str) -> np.ndarray:
    """沿轴 1 计算统计量并忽略 NaN：(N, M, 2, K) -> (N, 2, K)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if stat == "mean":
            return np.nanmean(x, axis=1)
        if stat == "median":
            return np.nanmedian(x, axis=1)
        if stat == "max":
            return np.nanmax(x, axis=1)
        if stat == "min":
            return np.nanmin(x, axis=1)
        if stat == "std":
            return np.nanstd(x, axis=1)
        m = re.fullmatch(r"(top|bot)(\d+)", stat)
        if m:
            # 同一文件同一特征的各值要么全为 NaN，要么全无 NaN；np.sort 将 NaN 排在末尾
            k = int(m.group(2))
            xs = np.sort(x, axis=1)
            part = xs[:, -k:] if m.group(1) == "top" else xs[:, :k]
            return part.mean(axis=1)
        if stat.startswith("q"):
            return np.nanpercentile(x, float(stat[1:]), axis=1)
    raise ValueError(f"未知统计量: {stat}")


def _car_reduce(cd: np.ndarray, stat: str) -> np.ndarray:
    """在 8 节车厢上聚合：(N, 8, 2, K) -> (N, 2, K)。adjmax / adjmin 基于相邻两节车厢的均值。"""
    if stat in ("adjmax", "adjmin"):
        adj = 0.5 * (cd[:, :-1] + cd[:, 1:])
        return _reduce(adj, "max" if stat == "adjmax" else "min")
    return _reduce(cd, stat)


def _prepare(cf: np.ndarray, baseline: np.ndarray | None):
    """减基线后返回 Side I (N,32,2,K)、Side II (N,32,2,K)、配对差值 (N,32,2,K)、
    车厢级差值 (N,8,2,K)（每节车厢 4 个轮对左右差值的均值，方向为 Side I - Side II）。"""
    x = np.asarray(cf, dtype=np.float64)
    if x.ndim == 4:
        x = x[None]
    if baseline is not None:
        x = x - np.asarray(baseline, dtype=np.float64)
    n = x.shape[0]
    shape = (n, -1, C.N_CHANNEL_TYPES, x.shape[-1])
    s1 = x[:, :, _SIDE_I].reshape(shape)
    s2 = x[:, :, _SIDE_II].reshape(shape)
    pd5 = x[:, :, _PAIR_I] - x[:, :, _PAIR_II]                  # (N, 8, 4, 2, K)
    pdiff = pd5.reshape(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cardiff = np.nanmean(pd5, axis=2)                         # (N, 8, 2, K)
    return s1, s2, pdiff, cardiff


def _block(arr: np.ndarray, prefix: str, stat: str) -> tuple[np.ndarray, list[str]]:
    names = [f"{prefix}_{ch}_{fn}_{stat}" for ch in CH_TAGS for fn in CHANNEL_FEATURE_NAMES]
    return arr.reshape(arr.shape[0], -1), names


def _car_block(cd: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names = [f"c{car}_{ch}_{fn}" for car in range(1, C.N_CARS + 1)
             for ch in CH_TAGS for fn in CHANNEL_FEATURE_NAMES]
    return cd.reshape(cd.shape[0], -1), names


def _side_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    need = tuple(dict.fromkeys(C.AGG_STATS + C.CONTRAST_STATS))
    return {s: _reduce(arr, s) for s in need}


def _check_car_features(car_features: str) -> None:
    if car_features not in C.CAR_FEATURE_SETS:
        raise ValueError(f"car_features 须为 {C.CAR_FEATURE_SETS} 之一，实际为 {car_features}")


def file_level(speeds, cf: np.ndarray, baseline: np.ndarray | None = None,
               car_features: str = "none") -> tuple[np.ndarray, list[str]]:
    """文件级特征：车速 + 两侧同侧聚合 + 两侧之差 + 配对差值聚合 [+ 车厢级特征]。"""
    _check_car_features(car_features)
    s1, s2, pdiff, cardiff = _prepare(cf, baseline)
    speeds = np.atleast_1d(np.asarray(speeds, dtype=np.float64))
    st1, st2 = _side_stats(s1), _side_stats(s2)

    cols, names = [speeds[:, None]], ["speed_mps"]

    def add(block):
        cols.append(block[0])
        names.extend(block[1])

    for tag, st in (("s1", st1), ("s2", st2)):
        for s in C.AGG_STATS:
            add(_block(st[s], tag, s))
    for s in C.CONTRAST_STATS:
        add(_block(st1[s] - st2[s], "d", s))
    for s in C.PAIR_STATS:
        add(_block(_reduce(pdiff, s), "p", s))
    if car_features in ("invariant", "full"):
        for s in C.CAR_STATS:
            add(_block(_car_reduce(cardiff, s), "k", s))
    if car_features == "full":
        add(_car_block(cardiff))
    return np.hstack(cols), names


def side_level(speeds, cf: np.ndarray, baseline: np.ndarray | None = None,
               car_features: str = "none"
               ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """单侧视图特征：每个文件拆为 Side I、Side II 两行。

    返回 X (2N, D)、特征名、所属文件下标 (2N,)、侧别 (2N,)。
    配对差值与车厢级差值统一为“本侧 - 对侧”方向。
    """
    _check_car_features(car_features)
    s1, s2, pdiff, cardiff = _prepare(cf, baseline)
    speeds = np.atleast_1d(np.asarray(speeds, dtype=np.float64))
    n = speeds.shape[0]
    st1, st2 = _side_stats(s1), _side_stats(s2)

    views, names = [], []
    for side, own, other, sign in (("Side I", st1, st2, 1.0), ("Side II", st2, st1, -1.0)):
        cols = [speeds[:, None], np.full((n, 1), float(side == "Side II"))]
        names = ["speed_mps", "is_side_II"]

        def add(block):
            cols.append(block[0])
            names.extend(block[1])

        for s in C.AGG_STATS:
            add(_block(own[s], "own", s))
        for s in C.CONTRAST_STATS:
            add(_block(own[s] - other[s], "d", s))
        for s in C.PAIR_STATS:
            add(_block(_reduce(sign * pdiff, s), "p", s))
        if car_features in ("invariant", "full"):
            for s in C.CAR_STATS:
                add(_block(_car_reduce(sign * cardiff, s), "k", s))
        if car_features == "full":
            add(_car_block(sign * cardiff))
        views.append(np.hstack(cols))

    x = np.stack(views, axis=1).reshape(2 * n, -1)
    file_idx = np.repeat(np.arange(n), 2)
    sides = np.array(list(C.SIDES) * n)
    return x, names, file_idx, sides


# ============================================================ 第三部分：车速残差化
def _log_speed(speeds) -> np.ndarray:
    return np.log(np.maximum(np.asarray(speeds, dtype=np.float64), C.MIN_SPEED_MPS))


def fit_speed_residual(x: np.ndarray, speeds, ref_mask, names: list[str],
                       min_rows: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """在参考样本上对每一列拟合 x = a + b*ln(v)，返回 (b, a)。"""
    x = np.asarray(x, dtype=np.float64)
    lv = _log_speed(speeds)[:, None]
    m = np.isfinite(x) & np.asarray(ref_mask, dtype=bool)[:, None]
    n = m.sum(axis=0)
    with np.errstate(all="ignore"):
        mx = np.where(m, lv, 0.0).sum(axis=0) / n
        my = np.where(m, x, 0.0).sum(axis=0) / n
        dx = np.where(m, lv - mx, 0.0)
        dy = np.where(m, x - my, 0.0)
        var = (dx * dx).sum(axis=0)
        slope = np.where((n >= min_rows) & (var > EPS), (dx * dy).sum(axis=0) / var, 0.0)
        intercept = my - slope * mx
    slope = np.nan_to_num(slope)
    intercept = np.nan_to_num(intercept)
    excluded = np.isin(np.asarray(names), RESIDUAL_EXCLUDE)
    slope[excluded] = 0.0
    intercept[excluded] = 0.0
    return slope, intercept


def apply_speed_residual(x: np.ndarray, speeds, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """减去车速拟合值：x - (a + b*ln(v))。NaN 保持不变。"""
    return np.asarray(x, dtype=np.float64) - (intercept + slope * _log_speed(speeds)[:, None])


# ============================================================ 自检
_CAR_SPECIFIC = re.compile(r"^c\d+_")


def _mirror_stat(stat: str) -> str | None:
    pairs = {"max": "min", "min": "max", "adjmax": "adjmin", "adjmin": "adjmax"}
    if stat in pairs:
        return pairs[stat]
    if stat.startswith("top"):
        return "bot" + stat[3:]
    if stat.startswith("bot"):
        return "top" + stat[3:]
    if stat.startswith("q"):
        return f"q{100 - int(stat[1:])}"
    return None


def _self_check() -> None:
    import time
    import data_io

    for s in C.CAR_STATS:
        assert s in ("adjmax", "adjmin") or C._valid_stat(s), s

    files = [C.TRAIN_DIR / "Train1.csv", C.TRAIN_DIR / "Train2.csv"]
    t0 = time.perf_counter()
    res = [extract_channel_features(*data_io.load_file(p)) for p in files]
    t1 = time.perf_counter()
    speeds = np.array([r[0] for r in res])
    cf = np.stack([r[1] for r in res])
    assert cf.shape == (2, C.N_CARS, C.N_POSITIONS, C.N_CHANNEL_TYPES, K)

    for label, baseline in (("no baseline", None), ("baseline", fit_baseline(cf))):
        fx, fn = file_level(speeds, cf, baseline, car_features="full")
        sx, sn, fidx, sides = side_level(speeds, cf, baseline, car_features="full")
        assert fx.shape == (2, len(fn)) and len(set(fn)) == len(fn)
        assert sx.shape == (4, len(sn)) and len(set(sn)) == len(sn)
        assert not np.isinf(fx).any() and not np.isinf(sx).any()

        # 单侧视图与文件级之间的对应关系
        fmaps = [dict(zip(fn, row)) for row in fx]
        for r in range(sx.shape[0]):
            fm, side = fmaps[fidx[r]], sides[r]
            tag, sign = ("s1", 1.0) if side == "Side I" else ("s2", -1.0)
            for name, v in zip(sn, sx[r]):
                if name.startswith("own_"):
                    exp = fm[f"{tag}_" + name[4:]]
                elif name.startswith("d_") or _CAR_SPECIFIC.match(name):
                    exp = sign * fm[name]
                elif name.startswith("p_") or name.startswith("k_"):
                    if side == "Side I":
                        exp = fm[name]
                    else:
                        base, stat = name.rsplit("_", 1)
                        mirror = _mirror_stat(stat)
                        key = f"{base}_{mirror}" if mirror else None
                        if key not in fm:
                            continue
                        exp = -fm[key]
                else:
                    continue
                assert np.allclose(v, exp, equal_nan=True), (label, side, name)

        # 各特征组合 = 完整特征去掉相应的车厢列
        for cs in C.CAR_FEATURE_SETS:
            fx_s, fn_s = file_level(speeds, cf, baseline, car_features=cs)
            sx_s, sn_s, _, _ = side_level(speeds, cf, baseline, car_features=cs)
            for full_x, full_n, sub_x, sub_n in ((fx, fn, fx_s, fn_s), (sx, sn, sx_s, sn_s)):
                keep = [i for i, n in enumerate(full_n)
                        if not ((cs == "none" and (n.startswith("k_") or _CAR_SPECIFIC.match(n)))
                                or (cs == "invariant" and _CAR_SPECIFIC.match(n)))]
                assert [full_n[i] for i in keep] == sub_n, (label, cs)
                assert np.allclose(full_x[:, keep], sub_x, equal_nan=True), (label, cs)
            if label == "baseline":
                print(f"[car_features={cs:<9}] file {fx_s.shape[1]} features, side {sx_s.shape[1]} features")
        print(f"[{label}] full: file {fx.shape}, side {sx.shape}, "
              f"NaN file/side = {int(np.isnan(fx).sum())}/{int(np.isnan(sx).sum())}")

    # 车速残差化：用已知关系 x = 3 + 2 ln(v) 的合成数据核验
    rng = np.random.default_rng(0)
    v = rng.uniform(2.0, 18.0, 50)
    a = np.column_stack([3 + 2 * np.log(v) + rng.normal(0, 0.01, 50), np.full(50, 7.0)])
    b_hat, a_hat = fit_speed_residual(a, v, np.ones(50, dtype=bool), ["feat", "speed_mps"])
    assert abs(b_hat[0] - 2) < 0.05 and abs(a_hat[0] - 3) < 0.1
    assert b_hat[1] == 0 and a_hat[1] == 0

    print("self-check passed")
    print(f"per-channel K    : {K}")
    print(f"extract 2 files  : {t1 - t0:.3f} s")


if __name__ == "__main__":
    _self_check()
