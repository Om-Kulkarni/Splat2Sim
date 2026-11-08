"""IMU helpers: unite separate gyro/accel streams, resample, clip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImuData:
    t_ns: np.ndarray  # (N,) int64
    gyro: np.ndarray  # (N,3) rad/s
    accel: np.ndarray  # (N,3) m/s^2

    def __len__(self) -> int:
        return len(self.t_ns)


def interp_columns(t_ref_ns: np.ndarray, t_ns: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of (N,3) samples at reference times (relative to t_ns[0] to keep precision)."""
    t0 = t_ns[0]
    tr = (np.asarray(t_ref_ns) - t0).astype(np.float64)
    ts = (np.asarray(t_ns) - t0).astype(np.float64)
    return np.stack([np.interp(tr, ts, x[:, k]) for k in range(x.shape[1])], axis=1)


def unite(gyro_t, gyro, accel_t, accel) -> ImuData:
    """Merge separate gyro and accel streams onto the timestamps of the faster one (linear interpolation of the
    other), limited to the time range both streams cover."""
    gyro_t, accel_t = np.asarray(gyro_t, np.int64), np.asarray(accel_t, np.int64)
    gyro, accel = np.asarray(gyro, np.float64), np.asarray(accel, np.float64)
    if len(gyro_t) < 2 or len(accel_t) < 2:
        raise ValueError("need at least 2 gyro and 2 accel samples to unite IMU streams")
    lo, hi = max(gyro_t[0], accel_t[0]), min(gyro_t[-1], accel_t[-1])
    if hi <= lo:
        raise ValueError("gyro and accel streams do not overlap in time")
    gyro_rate = (len(gyro_t) - 1) / (gyro_t[-1] - gyro_t[0])
    accel_rate = (len(accel_t) - 1) / (accel_t[-1] - accel_t[0])
    if gyro_rate >= accel_rate:
        sel = (gyro_t >= lo) & (gyro_t <= hi)
        t = gyro_t[sel]
        return ImuData(t, gyro[sel], interp_columns(t, accel_t, accel))
    sel = (accel_t >= lo) & (accel_t <= hi)
    t = accel_t[sel]
    return ImuData(t, interp_columns(t, gyro_t, gyro), accel[sel])


def sort_dedupe(imu: ImuData) -> ImuData:
    order = np.argsort(imu.t_ns, kind="stable")
    t = imu.t_ns[order]
    keep = np.ones(len(t), bool)
    keep[1:] = t[1:] != t[:-1]
    o = order[keep]
    return ImuData(imu.t_ns[o], imu.gyro[o], imu.accel[o])


def clip(imu: ImuData, t0_ns: int, t1_ns: int, margin_ns: int = 500_000_000) -> ImuData:
    m = (imu.t_ns >= t0_ns - margin_ns) & (imu.t_ns <= t1_ns + margin_ns)
    return ImuData(imu.t_ns[m], imu.gyro[m], imu.accel[m])


def rate_hz(t_ns: np.ndarray) -> float:
    return float((len(t_ns) - 1) / ((t_ns[-1] - t_ns[0]) / 1e9)) if len(t_ns) > 1 and t_ns[-1] > t_ns[0] else 0.0
