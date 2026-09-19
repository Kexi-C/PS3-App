#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Door 子系统 —— 单文件版 v2 (2026-09-19)，含目前全部已完成功能，供同事阅读/复现。
更新: 阈值 0.35; 评分函数与官方 Info Kit §4 逐条一致; 末段窗口均值判据。
==================================================================
依赖: numpy pandas scipy scikit-learn   (python 3.10+)

用法:
  python door_standalone.py eval    --train Train.csv --answer Train_Segments_Answer.csv
  python door_standalone.py predict --train Train.csv --answer Train_Segments_Answer.csv --test Test.csv --out door_predictions.csv
  python door_standalone.py timeline --train Train.csv --answer Train_Segments_Answer.csv --test Test.csv --seg 38 --out tl38.csv

整体逻辑（5 步）:
  1. 分割   : 连续流按时间间隔 >1 s 切成门周期（训练集 110 段边界与答案 100% 一致）。
  2. 逐点量 : 每段按 20 ms 采样，用 Savitzky-Golay 求 0/1/2/3 阶导数（局部泰勒系数），
              得到电流/电压/速度/加速度/功率/力矩转速比/电机模型残差等通道。
  3. 正常模板: 只用"正常"段，按门位置(10 单位一桶)统计每个通道的均值和标准差；
              任意样本点的 z = (x - 均值[位置]) / 标准差[位置]  → 每个采样点(可重采样到每 1 ms)都有检测值。
  4. 段级判据: 把逐点 z 在巡航/减速窗口上取 p90、最长连续超限长度、均值等，共 10 个判据
              (电流、电流斜率、电压残差、力矩/转速比、能量 五个方面)，log1p 后进 L2 逻辑回归。
  5. 评分   : IoU 加权 F1（与 Door_Subsystem_Info_Kit.md §4 官方公式一致）。

防过拟合: 模板/分类器只在训练折内拟合; 分块时间交叉验证; 10 个特征 + 强正则;
          正常段做物理一致的时间伸缩增广(模拟快/慢门), 加速段惯性电流与到位前凸包不参与阻力评分。
"""
from __future__ import annotations
import argparse, sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------------- 常量
COL_I, COL_V, COL_E, COL_P = "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force", "Door leaf position"
COL_OPENING = "Door is opening"
DT = 0.02                      # 采样周期 20 ms
POS_FULL = 700.0               # 全行程
E_PER_SPEED = 4.73             # 实测: 反电动势列 = 4.73 * |dP/dt|  → 该列是速度代理
CRUISE_LO, CRUISE_HI, DECEL_HI = 140.0, 560.0, 640.0   # 相位边界(按进度 prog)
BIN_W = 10.0                   # 模板位置桶宽
ABN, NORM = "Abnormal resistance", "Normal"
CHANNELS = ["I_s", "dI", "d2I", "V_s", "dV", "V_res", "E", "speed", "accel", "I_over_E", "V_over_E", "power"]
MIN_STD = {"I_s": 15, "dI": 300, "d2I": 5000, "V_s": 100, "dV": 1000, "V_res": 60, "E": 30, "speed": 8,
           "accel": 150, "I_over_E": 0.01, "V_over_E": 0.2, "power": 100}
# 最终使用的 10 个判据(+操作类型标志)
COMPACT = ["op_is_open",
           "cur_I_s_cruise_p90",        # 电流: 巡航段 z 的 90 分位
           "cur_I_s_steady_run_gt",     # 电流: 140-640 内 z>3 的最长连续样本数
           "cur_I_s_decel_p90",         # 电流: 减速段(560-640) z 的 p90 → 后段阻力
           "cur_dI_steady_p90",         # 电流一阶导: 阻力突变
           "cur_end_mean_excess",       # 电流: 640-690 窗口均值超出模板的 σ 数(末段安全网)
           "vol_V_res_steady_p90",      # 电压: 电机模型残差 V-(R I+Ke E+c) 的 z p90 (速度不变)
           "vol_V_res_cruise_absmean",  # 电压: 巡航段 |残差| 均值
           "rat_I_over_E_steady_p90",   # 比值: 力矩/转速 (I/E) 的 z p90
           "ene_power_steady_p90",      # 能量: 瞬时功率 I·V 的 z p90
           "ene_energy_per_unit"]       # 能量: ∫I·V dt / 实际位移


# ----------------------------------------------------------------------------- 1. 读取与分割
def parse_dt(s: str) -> pd.Timestamp:
    """'2023-7-5-0-0-23-999' → Timestamp (最后一段是毫秒)"""
    p = [int(x) for x in str(s).split("-")]
    return pd.Timestamp(p[0], p[1], p[2], p[3], p[4], p[5]) + pd.Timedelta(milliseconds=p[6])


def fmt_dt(t: pd.Timestamp) -> str:
    return f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}-{t.microsecond // 1000}"


def load_stream(path) -> pd.DataFrame:
    df = pd.read_csv(path); df["t"] = df["Datetime"].map(parse_dt)
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def split_segments(df: pd.DataFrame, gap_s: float = 1.0, min_rows: int = 20) -> list[pd.DataFrame]:
    """时间间隔 > gap_s 或 开/关标志翻转 → 新段"""
    gap = df["t"].diff().dt.total_seconds().fillna(0).values > gap_s
    flag = df[COL_OPENING].values.astype(int); flip = np.r_[False, flag[1:] != flag[:-1]]
    sid = np.cumsum(gap | flip)
    return [g.reset_index(drop=True) for _, g in df.groupby(sid, sort=True) if len(g) >= min_rows]


def op_of(seg) -> str:
    return "Open" if int(round(seg[COL_OPENING].mean())) == 1 else "Close"


# ----------------------------------------------------------------------------- 2. 逐点导出量 (泰勒各阶)
def _sg(x, deriv, win=9, order=3):
    """Savitzky-Golay: 9 点(180 ms)三次多项式局部拟合, deriv = 0/1/2/3 阶导数"""
    x = np.asarray(x, float); n = len(x)
    if n < win:
        win = n if n % 2 else n - 1
        if win <= order: return np.zeros(n) if deriv else x
    return savgol_filter(x, win, order, deriv=deriv, delta=DT, mode="interp")


def derive(seg: pd.DataFrame) -> pd.DataFrame:
    op = op_of(seg); I, V, E, P = (seg[c].values.astype(float) for c in (COL_I, COL_V, COL_E, COL_P))
    prog = P if op == "Open" else POS_FULL - P          # 进度: 两种操作都从 0 走到 700
    d = pd.DataFrame({"t": np.arange(len(seg)) * DT, "I": I, "V": V, "E": E, "P": P, "prog": prog})
    d["speed"] = np.abs(_sg(P, 1)); d["accel"] = _sg(prog, 2); d["jerk"] = _sg(prog, 3)   # 位置 1/2/3 阶
    d["I_s"], d["dI"], d["d2I"] = _sg(I, 0), _sg(I, 1), _sg(I, 2)                          # 电流 0/1/2 阶
    d["V_s"], d["dV"] = _sg(V, 0), _sg(V, 1)
    d["E_over_speed"] = E / np.maximum(d["speed"], 1) / E_PER_SPEED   # ≈1: 传感器自洽(唯一天然无量纲量)
    d["I_over_E"] = I / np.maximum(E, 50)                             # 力矩/转速
    d["V_over_E"] = V / np.maximum(E, 50)
    d["power"] = I * V / 1e3; d["energy"] = np.cumsum(d["power"]) * DT
    # 相位: 超过 705 仍在动 = overrun(传感器/标定异常, 如测试 #33 位置到 807), 不参与阻力评分
    moving = d["speed"].values > 5
    d["overrun"] = (prog > POS_FULL + 5) & moving
    d["moving"] = moving; d.attrs["op"] = op
    return d


def resample_ms(d: pd.DataFrame, step_ms=1, cols=None) -> pd.DataFrame:
    """线性重采样到整数毫秒网格 → 每个 ms 都有值"""
    t = d["t"].values; grid = np.arange(0, int(round(t[-1] * 1000)) + 1, step_ms) / 1000
    cols = cols or [c for c in d.columns if d[c].dtype.kind in "fi" and c != "t"]
    return pd.DataFrame({"t_ms": (grid * 1000).round().astype(int), **{c: np.interp(grid, t, d[c].values.astype(float)) for c in cols}})


# ----------------------------------------------------------------------------- 3. 正常模板 + 电机模型
class NormalTemplate:
    """按操作类型、按门位置桶存正常段各通道的均值/标准差; 另拟合 V ≈ R·I + Ke·E + c"""
    def __init__(self): self.tables, self.motor = {}, {}

    @staticmethod
    def _bins(prog): return (np.minimum(prog, POS_FULL) // BIN_W).astype(int)

    def add_residual(self, d):
        R, Ke, c = self.motor[d.attrs["op"]]
        d["V_res"] = d["V"].values - (R * d["I"].values + Ke * d["E"].values + c); return d

    def fit(self, derived: list[pd.DataFrame]):
        ops = [d.attrs["op"] for d in derived]
        for op in set(ops):
            big = pd.concat([d for d, o in zip(derived, ops) if o == op], ignore_index=True)
            m = (big["speed"].values > 50) & (big["V"].values < 10400)          # 去掉 PWM 饱和点
            A = np.c_[big["I"].values[m], big["E"].values[m], np.ones(m.sum())]
            self.motor[op] = np.linalg.lstsq(A, big["V"].values[m], rcond=None)[0]
        derived = [self.add_residual(d.copy()) for d in derived]
        for op in set(ops):
            big = pd.concat([d for d, o in zip(derived, ops) if o == op], ignore_index=True)
            g = big[CHANNELS].groupby(self._bins(big["prog"].values))
            mean, std = g.mean(), g.std().fillna(0)
            for c in CHANNELS: std[c] = np.maximum(std[c], np.maximum(MIN_STD[c], 0.05 * mean[c].abs()))
            self.tables[op] = (mean, std)
        return self

    def zscores(self, d: pd.DataFrame) -> pd.DataFrame:
        if "V_res" not in d: self.add_residual(d)
        mean, std = self.tables[d.attrs["op"]]
        pos = np.searchsorted(mean.index.values, self._bins(d["prog"].values)).clip(0, len(mean) - 1)
        return pd.DataFrame({f"z_{c}": (d[c].values - mean[c].values[pos]) / std[c].values[pos] for c in CHANNELS})


# ----------------------------------------------------------------------------- 4. 段级判据
def _run_len(mask):
    best = cur = 0
    for m in mask: cur = cur + 1 if m else 0; best = max(best, cur)
    return best


def _agg(z, prefix, out, thr=3.0):
    z = np.asarray(z, float)
    if len(z) == 0:
        for k in ("max", "mean", "p90", "frac_gt", "run_gt"): out[f"{prefix}_{k}"] = 0.0
        return
    out[f"{prefix}_max"] = z.max(); out[f"{prefix}_mean"] = z.mean(); out[f"{prefix}_p90"] = np.percentile(z, 90)
    out[f"{prefix}_frac_gt"] = np.mean(z > thr); out[f"{prefix}_run_gt"] = float(_run_len(z > thr))


def segment_features(seg: pd.DataFrame, tpl: NormalTemplate) -> dict:
    d = derive(seg); z = tpl.zscores(d); op = d.attrs["op"]
    prog, moving, ok = d["prog"].values, d["moving"].values, ~d["overrun"].values
    cruise = (prog >= CRUISE_LO) & (prog <= CRUISE_HI) & ok
    decel = (prog > CRUISE_HI) & (prog <= DECEL_HI) & ok
    steady = (prog >= CRUISE_LO) & (prog <= DECEL_HI) & ok     # 阻力评分窗口(加速段惯性电流、末段凸包不计入)
    travel = moving & (prog < POS_FULL - 5) & ok
    f = {"op_is_open": float(op == "Open")}
    for ch in ("I_s", "dI", "d2I"):
        for name, m in (("cruise", cruise), ("decel", decel), ("steady", steady), ("travel", travel)): _agg(z[f"z_{ch}"].values[m], f"cur_{ch}_{name}", f)
    for ch in ("V_s", "V_res"):
        for name, m in (("cruise", cruise), ("steady", steady)): _agg(z[f"z_{ch}"].values[m], f"vol_{ch}_{name}", f)
    f["vol_V_res_cruise_absmean"] = float(np.abs(d["V_res"].values[cruise]).mean()) if cruise.any() else 0.0
    for ch in ("I_over_E", "V_over_E"): _agg(z[f"z_{ch}"].values[steady], f"rat_{ch}_steady", f)
    _agg(z["z_power"].values[steady], "ene_power_steady", f)
    f["ene_energy_per_unit"] = float(d["energy"].values[-1]) / (abs(prog[-1] - prog[0]) or 1.0)
    # 末段 640-690: 到位前电流凸包位置随门速前移, 逐点 z 不可靠 → 只比较窗口均值
    endw = (prog > 640) & (prog <= 690) & moving & ok
    if endw.any():
        mean, std = tpl.tables[op]; pos = np.searchsorted(mean.index.values, tpl._bins(np.arange(645, 690, 10.0))).clip(0, len(mean) - 1)
        f["cur_end_mean_excess"] = float((d["I"].values[endw].mean() - mean["I_s"].values[pos].mean()) / max(std["I_s"].values[pos].mean(), 15))
    else: f["cur_end_mean_excess"] = 0.0
    f["kin_n_rows"] = float(len(d)); f["kin_overshoot"] = float(max(0, prog.max() - POS_FULL)); f["q_overrun_rows"] = float((~ok).sum())
    return f


# ----------------------------------------------------------------------------- 增广: 物理一致的快/慢门
def warp(seg: pd.DataFrame, factor: float, ke: float = 3.8) -> pd.DataFrame:
    """时间伸缩 factor(<1 更快): E∝速度按 1/f, V 按 Ke·ΔE 调整, 惯性电流按 1/f², 摩擦电流不变"""
    n = len(seg); m = max(int(round(n * factor)), 10); xo, xn = np.linspace(0, 1, n), np.linspace(0, 1, m)
    out = seg.iloc[np.clip((xn * (n - 1)).round().astype(int), 0, n - 1)].copy().reset_index(drop=True)
    p = np.interp(xn, xo, seg[COL_P].values.astype(float)); e_old = np.interp(xn, xo, seg[COL_E].values.astype(float)); e_new = e_old / factor
    i_old = np.interp(xn, xo, seg[COL_I].values.astype(float)); acc = np.gradient(np.gradient(p, DT * factor), DT * factor)
    inertial = np.clip(0.4 * np.abs(acc) / max(np.abs(acc).max(), 1) * i_old, 0, None)
    out[COL_P] = np.round(p); out[COL_E] = np.round(e_new); out[COL_I] = np.round(i_old - inertial + inertial / factor ** 2)
    out[COL_V] = np.round(np.clip(np.interp(xn, xo, seg[COL_V].values.astype(float)) + ke * (e_new - e_old), 0, 10500) / 100) * 100
    out["t"] = seg["t"].iloc[0] + pd.to_timedelta(np.arange(m) * DT, unit="s"); return out


# ----------------------------------------------------------------------------- 5. 模型
class DoorModel:
    def __init__(self, features=COMPACT, C=0.3, augment=(0.85, 0.92, 1.08), thr=0.35):
        self.features, self.C, self.augment, self.thr = list(features), C, augment, thr
        self.tpl, self.clf = NormalTemplate(), None

    @staticmethod
    def _log(X):  # 非负特征 log1p: 压缩重尾, 使边界靠近紧凑的正常簇
        X = np.asarray(X, float).copy(); pos = X.min(axis=0) >= 0; X[:, pos] = np.log1p(X[:, pos]); return X

    def fit(self, segs, labels):
        segs, labels = list(segs), list(labels)
        for s, l in list(zip(segs, labels)):
            if l == NORM: segs += [warp(s, f) for f in self.augment]; labels += [NORM] * len(self.augment)
        self.tpl.fit([derive(s) for s, l in zip(segs, labels) if l == NORM])          # 模板只用正常段
        X = self._log(self.featurize(segs)[self.features].values); y = np.array([l == ABN for l in labels], int)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)).fit(X, y)
        return self

    def featurize(self, segs): return pd.DataFrame([segment_features(s, self.tpl) for s in segs])
    def predict_proba(self, segs): return self.clf.predict_proba(self._log(self.featurize(segs)[self.features].values))[:, 1]
    def predict(self, segs): return [ABN if p >= self.thr else NORM for p in self.predict_proba(segs)]


def blocked_cv(segs, labels, k=5, **kw):
    """连续时间块交叉验证(不打乱): 模板与分类器均只在训练块上拟合"""
    n = len(segs); edges = np.linspace(0, n, k + 1).astype(int); proba = np.zeros(n)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[np.arange(0, a), np.arange(b, n)]; te = np.arange(a, b)
        proba[te] = DoorModel(**kw).fit([segs[i] for i in tr], [labels[i] for i in tr]).predict_proba([segs[i] for i in te])
    return proba


# ----------------------------------------------------------------------------- 6. 评分: IoU 加权 F1
def iou_weighted_f1(pred: pd.DataFrame, true: pd.DataFrame) -> dict:
    """pred: start_time,end_time,prediction ; true: start_time,end_time,status
    官方规则(Door_Subsystem_Info_Kit.md §4, 已核对): 预测段只能匹配同标签的真实段; IoU>0 的候选对
    按 IoU 从高到低一对一贪心匹配; soft_recall = ΣIoU/真实段数, soft_precision = ΣIoU/预测段数,
    score = 调和平均。标签错的预测 = 一个漏检 + 一个误报。"""
    ps = pred.start_time.map(lambda s: parse_dt(s).value / 1e9).values; pe = pred.end_time.map(lambda s: parse_dt(s).value / 1e9).values
    ts = true.start_time.map(lambda s: parse_dt(s).value / 1e9).values; te = true.end_time.map(lambda s: parse_dt(s).value / 1e9).values
    if len(pred) == 0 or len(true) == 0: return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    inter = np.maximum(0, np.minimum(te[:, None], pe[None]) - np.maximum(ts[:, None], ps[None]))
    M = inter / ((te - ts)[:, None] + (pe - ps)[None] - inter)
    same = np.asarray(true.status.astype(str), dtype=object)[:, None] == np.asarray(pred.prediction.astype(str), dtype=object)[None, :]
    M = np.where(same, M, 0.0)                                   # 标签不同 → 不可匹配
    used_t, used_p, w = set(), set(), 0.0
    for i, j in np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]:
        if M[i, j] <= 0: break
        if i in used_t or j in used_p: continue
        used_t.add(i); used_p.add(j); w += M[i, j]
    P, R = w / len(pred), w / len(true)
    return {"f1": 2 * P * R / (P + R) if P + R else 0.0, "precision": P, "recall": R}


# ----------------------------------------------------------------------------- 7. 端到端
def predict_stream(test_csv, model: DoorModel) -> tuple[pd.DataFrame, list]:
    segs = split_segments(load_stream(test_csv)); p = model.predict_proba(segs)
    rows = [{"start_time": fmt_dt(s["t"].iloc[0]), "end_time": fmt_dt(s["t"].iloc[-1]), "prediction": ABN if pi >= model.thr else NORM,
             "operation": op_of(s), "p_abnormal": round(float(pi), 4), "n_rows": len(s)} for s, pi in zip(segs, p)]
    return pd.DataFrame(rows), segs


def anomaly_timeline(seg, model: DoorModel, step_ms=1) -> pd.DataFrame:
    """单个周期的逐毫秒检测轨迹: 关键通道 z 分数 + 综合分(max|z| 的 5 点滑动均值)"""
    d = derive(seg); z = model.tpl.zscores(d); keep = ["z_I_s", "z_dI", "z_V_s", "z_V_res", "z_I_over_E", "z_power", "z_speed"]
    tl = pd.concat([d[["t", "P", "prog", "I", "V", "E", "speed"]], z[keep]], axis=1)
    tl["score"] = pd.Series(np.abs(tl[keep].values).max(axis=1)).rolling(5, center=True, min_periods=1).mean().values
    return resample_ms(tl, step_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["eval", "predict", "timeline"]); ap.add_argument("--train", required=True); ap.add_argument("--answer", required=True)
    ap.add_argument("--test"); ap.add_argument("--out", default="door_predictions.csv"); ap.add_argument("--seg", type=int, default=1); ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    segs = split_segments(load_stream(a.train)); ans = pd.read_csv(a.answer); labels = ans.status.tolist()
    assert len(segs) == len(ans), f"分割得到 {len(segs)} 段, 答案 {len(ans)} 段"
    if a.cmd == "eval":
        proba = blocked_cv(segs, labels, a.folds); y = np.array([l == ABN for l in labels])
        pred = proba >= 0.35; tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
        f1 = 2 * tp / (2 * tp + fp + fn); print(f"blocked {a.folds}-fold CV: F1={f1:.4f}  FP={fp}  FN={fn}  normal max p={proba[~y].max():.3f}  abnormal min p={proba[y].min():.3f}")
        sub = pd.DataFrame({"start_time": ans.start_time, "end_time": ans.end_time, "prediction": np.where(pred, ABN, NORM)})
        print("IoU-weighted F1 on train (CV predictions, true boundaries):", {k: round(v, 4) for k, v in iou_weighted_f1(sub, ans).items()})
        return
    model = DoorModel().fit(segs, labels)
    pred, tsegs = predict_stream(a.test, model)
    if a.cmd == "predict":
        pred[["start_time", "end_time", "prediction"]].to_csv(a.out, index=False)
        print(pred.to_string()); print(f"\n{(pred.prediction == ABN).sum()}/{len(pred)} abnormal → {a.out}")
    else:
        tl = anomaly_timeline(tsegs[a.seg - 1], model); tl.to_csv(a.out, index=False)
        print(tl.iloc[::250].round(2).to_string(index=False)); print(f"\n{len(tl)} rows (1 ms grid) → {a.out}")


if __name__ == "__main__":
    main()
