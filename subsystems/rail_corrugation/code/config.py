"""Subsystem 3 (Rail Corrugation) 全局配置。

所有路径、物理常量、数据布局与实验参数集中定义于此，其余模块只从本文件读取。
路径可通过环境变量覆盖：RAIL_DATA_DIR、RAIL_CACHE_DIR。
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

# ============================================================ 路径
REPO_ROOT = Path(__file__).resolve().parents[1]      # Subsystem3/
WORKSPACE_ROOT = REPO_ROOT.parent                     # 两个仓库的共同上级目录

DATA_DIR = Path(os.environ.get(
    "RAIL_DATA_DIR",
    WORKSPACE_ROOT / "NebulaX-Hackathon-ProblemStatement" / "PS3"
    / "02_Datasets" / "Rail_Corrugation",
)).resolve()
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
LABELS_CSV = DATA_DIR / "Train_Labels.csv"

# 缓存置于仓库之外，不纳入版本控制
CACHE_DIR = Path(os.environ.get(
    "RAIL_CACHE_DIR", WORKSPACE_ROOT / "Subsystem3_cache"
)).resolve()
RAW_CACHE_DIR = CACHE_DIR / "raw_npy"
CHANNEL_FEATURES_TRAIN = CACHE_DIR / "channel_features_train.npz"   # 逐轴箱特征（train.py 使用）
CHANNEL_FEATURES_TEST = CACHE_DIR / "channel_features_test.npz"
FEATURES_TRAIN = CACHE_DIR / "features_train.pkl"                   # 聚合特征表（eda.py 使用）
FEATURES_TEST = CACHE_DIR / "features_test.pkl"
OOF_PREDICTIONS = CACHE_DIR / "oof_predictions.csv"

MODEL_DIR = REPO_ROOT / "model"
MODEL_FILE = MODEL_DIR / "rail_model.joblib"

# ============================================================ 采集参数
FS_HZ = 10_000                      # 采样频率
N_SAMPLES = 10_000                  # 每文件采样点数（不含表头）
DURATION_S = N_SAMPLES / FS_HZ      # 1 s

N_TEETH = 90                        # 测速齿轮齿数
TRANSITIONS_PER_REV = 2 * N_TEETH   # 每齿进入、离开检测点各产生一次 0/1 跳变
WHEEL_DIAMETER_M = 0.85
WHEEL_CIRCUMFERENCE_M = math.pi * WHEEL_DIAMETER_M

# ============================================================ 数据布局
# 第 1 列为转速脉冲；第 2-129 列按 车(1..8) -> 位置(1..8) -> 通道(振动, 冲击) 顺序排列
N_CARS = 8
N_POSITIONS = 8
CHANNEL_TYPES = ("vibration", "shock")
N_CHANNEL_TYPES = len(CHANNEL_TYPES)
N_SIGNAL_COLUMNS = N_CARS * N_POSITIONS * N_CHANNEL_TYPES   # 128
N_COLUMNS = 1 + N_SIGNAL_COLUMNS                            # 129
SPEED_COL = 0                       # 0 起始列下标
SIGNAL_COL_START = 1
DTYPE = "float32"

# 位置编号为 1 起始；*_POS_IDX 为对应的 0 起始数组下标
SIDES = ("Side I", "Side II")
SIDE_I_POSITIONS = (1, 3, 5, 7)
SIDE_II_POSITIONS = (2, 4, 6, 8)
SIDE_I_POS_IDX = tuple(p - 1 for p in SIDE_I_POSITIONS)
SIDE_II_POS_IDX = tuple(p - 1 for p in SIDE_II_POSITIONS)

# 同一轮对左右两侧的位置编号 (Side I 位置, Side II 位置)
# 假设：相邻编号属于同一轮对；须以 03_References/Rail_Corrugation/images 中的位置分布图核对
AXLE_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8))


def signal_column(car: int, position: int, channel_type: str) -> int:
    """返回指定信号的 0 起始列下标；car 与 position 为 1 起始编号。"""
    if not (1 <= car <= N_CARS and 1 <= position <= N_POSITIONS):
        raise ValueError(f"car/position 越界: car={car}, position={position}")
    k = CHANNEL_TYPES.index(channel_type)
    return (SIGNAL_COL_START
            + (car - 1) * N_POSITIONS * N_CHANNEL_TYPES
            + (position - 1) * N_CHANNEL_TYPES
            + k)


# ============================================================ 标签与提交格式
CLASS_NAMES = ("Normal", "Side I", "Side II")
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
LABELS_FILE_COL = "filename"
LABELS_LABEL_COL = "label"

SUBMISSION_FILENAME = "rail_predictions.csv"
SUBMISSION_FILE_COL = "file_id"
SUBMISSION_PRED_COL = "prediction"

# ============================================================ 特征参数
WELCH_NPERSEG = 2048                 # 频率分辨率约 4.88 Hz
WELCH_NOVERLAP = WELCH_NPERSEG // 2
FREQ_BANDS_HZ = (
    (10, 50), (50, 100), (100, 200), (200, 400),
    (400, 800), (800, 1600), (1600, 3200), (3200, 5000),
)
# 波长域频带（半倍程，2-64 cm）：lambda = v / f，用于消除车速对特征频率的影响
WAVELENGTH_BANDS_M = (
    (0.02, 0.0283), (0.0283, 0.04), (0.04, 0.0566), (0.0566, 0.08), (0.08, 0.1131),
    (0.1131, 0.16), (0.16, 0.2263), (0.2263, 0.32), (0.32, 0.4525), (0.4525, 0.64),
)
MIN_SPEED_MPS = 1.0                  # 低于此车速时波长域特征视为无定义

# 聚合统计量：mean / median / max / min / std；qXX 为第 XX 百分位数；
# topN / botN 为最大 / 最小 N 个值的均值
AGG_STATS = ("mean", "median", "max", "std")                        # 同侧 32 个轴箱上的聚合
CONTRAST_STATS = ("mean", "median", "max", "top3")                  # 两侧之差（本侧 - 对侧）
PAIR_STATS = ("max", "min", "top3", "bot3", "q90", "q10")           # 32 个轮对左右配对差值上的聚合
# 车厢级特征：每节车厢 4 个轮对左右差值的均值，再在 8 节车厢上聚合
# adjmax / adjmin 为相邻两节车厢均值中的最大 / 最小值
CAR_STATS = ("max", "top2", "min", "bot2", "adjmax", "adjmin")
CAR_FEATURE_SETS = ("none", "invariant", "full")   # 不加 / 仅位置无关 / 位置无关 + 逐车厢

# ============================================================ 建模参数
LOW_SPEED_RULE_MPS = 1.0             # 低于此车速的文件直接判为 Normal，不进入模型
RANDOM_SEED = 42
CV_N_SPLITS = 5
CV_N_REPEATS = 3
N_JOBS = -1


def _valid_stat(s: str) -> bool:
    return (s in ("mean", "median", "max", "min", "std")
            or re.fullmatch(r"q\d{1,2}", s) is not None
            or re.fullmatch(r"(top|bot)\d+", s) is not None)


def _self_check() -> None:
    assert N_COLUMNS == 129
    assert signal_column(1, 1, "vibration") == 1
    assert signal_column(1, 1, "shock") == 2
    assert signal_column(1, 2, "vibration") == 3
    assert signal_column(2, 1, "vibration") == 17
    assert signal_column(8, 8, "shock") == N_COLUMNS - 1
    assert set(SIDE_I_POS_IDX) | set(SIDE_II_POS_IDX) == set(range(N_POSITIONS))
    assert all(a in SIDE_I_POSITIONS and b in SIDE_II_POSITIONS for a, b in AXLE_PAIRS)
    assert sorted(p for pair in AXLE_PAIRS for p in pair) == list(range(1, N_POSITIONS + 1))
    assert all(_valid_stat(s) for s in AGG_STATS + CONTRAST_STATS + PAIR_STATS)
    assert all(lo < hi for lo, hi in WAVELENGTH_BANDS_M)


if __name__ == "__main__":
    _self_check()
    print("self-check passed")
    for name, path in [
        ("DATA_DIR", DATA_DIR), ("TRAIN_DIR", TRAIN_DIR), ("TEST_DIR", TEST_DIR),
        ("LABELS_CSV", LABELS_CSV), ("CACHE_DIR", CACHE_DIR), ("MODEL_DIR", MODEL_DIR),
    ]:
        print(f"{name:<11} exists={str(path.exists()):<5} {path.relative_to(WORKSPACE_ROOT)}")
