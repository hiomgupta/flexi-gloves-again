"""
Wearable Rule-based Mood Detection Module

Implements deterministic mood classification using only MAX30100 (HR, PPG, IBI),
MPU6050 (ACC, optional GYRO) and BMP280 (TEMP).

Design notes:
- Computes baselines from a calibration period.
- Computes features in 8-second windows.
- Motion gating rejects classification when activity detected.
- Classifier implements the EXACT rule set provided by the user.

Functions required by the specification are implemented with the exact signatures.

Author: Copilot (GPT-5 mini)
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

EPS = 1e-8


class Baseline:
    """
    Holds baseline values computed during a calm calibration period.

    Attributes:
        HR_base (float)
        RMSSD_base (float)
        PAV_base (float)
        TEMP_base (float)
        ACC_sd_base (float)
        GYRO_base (float)
    """

    def __init__(self,
                 HR_base: float = np.nan,
                 RMSSD_base: float = np.nan,
                 PAV_base: float = np.nan,
                 TEMP_base: float = np.nan,
                 ACC_sd_base: float = np.nan,
                 GYRO_base: float = np.nan):
        self.HR_base = HR_base
        self.RMSSD_base = RMSSD_base
        self.PAV_base = PAV_base
        self.TEMP_base = TEMP_base
        self.ACC_sd_base = ACC_sd_base
        self.GYRO_base = GYRO_base

    def as_dict(self) -> Dict[str, float]:
        return {
            "HR_base": self.HR_base,
            "RMSSD_base": self.RMSSD_base,
            "PAV_base": self.PAV_base,
            "TEMP_base": self.TEMP_base,
            "ACC_sd_base": self.ACC_sd_base,
            "GYRO_base": self.GYRO_base,
        }


def _safe_median(x):
    try:
        return float(np.nanmedian(np.asarray(x)))
    except Exception:
        return np.nan


def _compute_rmssd(ibi_series: np.ndarray) -> float:
    """
    Compute RMSSD from an IBI series (inter-beat intervals). IBI units should be ms or seconds,
    but RMSSD is relative since ratios are used later. If ibi_series has fewer than 2 points,
    returns NaN.
    """
    ibi_series = np.asarray(ibi_series)
    if ibi_series.size < 2:
        return np.nan
    diffs = np.diff(ibi_series)
    # Guard against corrupted diffs
    if diffs.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(diffs ** 2)))


def _compute_ppg_pav(ppg_signal: np.ndarray, fs: Optional[float] = None) -> float:
    """
    Compute mean peak-to-trough amplitude (PAV) from a raw PPG waveform.
    Uses scipy.signal.find_peaks to find peaks and troughs. If not enough peaks/troughs,
    returns NaN.

    Args:
        ppg_signal: 1D numpy array of PPG samples.
        fs: optional sampling frequency (not required for algorithm here).
    Returns:
        mean peak-to-trough amplitude (float) or NaN when insufficient data.
    """
    signal = np.asarray(ppg_signal)
    if signal.size < 3:
        return np.nan

    # Find peaks
    peaks, _ = find_peaks(signal, distance=1)
    # Find troughs by inverting signal
    troughs, _ = find_peaks(-signal, distance=1)

    if peaks.size == 0 or troughs.size == 0:
        return np.nan

    # For each peak, find nearest trough before the peak
    amps = []
    troughs_sorted = np.sort(troughs)
    for p in peaks:
        # find troughs that occur before p
        before = troughs_sorted[troughs_sorted < p]
        if before.size == 0:
            continue
        t = before[-1]
        amp = signal[p] - signal[t]
        amps.append(amp)

    if len(amps) == 0:
        return np.nan

    return float(np.nanmean(amps))


def compute_baselines(hr_series,
                      ibi_series,
                      ppg_series,
                      acc_series,
                      temp_series,
                      gyro_series: Optional[np.ndarray] = None) -> Baseline:
    """
    Compute baselines during a calm calibration period (3-5 minutes).

    Parameters expected to be array-like (numpy arrays, lists, or pandas Series).

    Returns:
        Baseline object with median values as specified.

    Implementation notes:
    - HR_base: median of hr_series
    - RMSSD_base: RMSSD computed on ibi_series (single value) — returned as median (single value)
    - PAV_base: computed from ppg_series using peak-to-trough amplitudes
    - TEMP_base: median of temp_series
    - ACC_sd_base: compute acceleration magnitude then std over the full calibration window
    - GYRO_base: median gyro energy (if provided) computed as median(sum(axis=1, gyro^2))

    The function prefers to operate on full series; if the user provides time-indexed pandas Series
    this will also work. For brevity and robustness we compute the medians from the whole calibration
    arrays (this aligns with the requirement to use medians over the rest period).
    """
    # HR_base
    HR_base = _safe_median(hr_series)

    # RMSSD_base: compute RMSSD across the entire ibi_series
    try:
        ibi_arr = np.asarray(ibi_series)
        RMSSD_base = _compute_rmssd(ibi_arr)
    except Exception:
        RMSSD_base = np.nan

    # PAV_base
    try:
        ppg_arr = np.asarray(ppg_series)
        PAV_base = _compute_ppg_pav(ppg_arr)
    except Exception:
        PAV_base = np.nan

    # TEMP_base
    TEMP_base = _safe_median(temp_series)

    # ACC_sd_base: use the quietest portion of the calibration window (bottom X percentile)
    # This reduces bias if the user fidgets during calibration.
    try:
        acc = np.asarray(acc_series)
        if acc.ndim == 1:
            acc_mag = acc
        else:
            acc_mag = np.sqrt(np.sum(acc ** 2, axis=1)).astype(float)

        # If calibration has very few samples, fall back to overall std
        if acc_mag.size < 10:
            ACC_sd_base = float(np.nanstd(acc_mag))
        else:
            # Split into segments and compute std per segment
            # Choose ~60 segments if possible to approximate 1s windows over multi-minute data
            n_segments = min(max(acc_mag.size // 50, 4), 200)
            seg_len = max(1, acc_mag.size // n_segments)
            seg_stds = []
            for i in range(0, acc_mag.size, seg_len):
                seg = acc_mag[i:i + seg_len]
                if seg.size > 0:
                    seg_stds.append(float(np.nanstd(seg)))
            seg_stds = np.asarray(seg_stds)
            if seg_stds.size == 0:
                ACC_sd_base = float(np.nanstd(acc_mag))
            else:
                # take bottom 5th percentile of segment stds (configurable)
                pct = 5.0
                k = max(1, int(np.floor(len(seg_stds) * (pct / 100.0))))
                lower = np.sort(seg_stds)[:k]
                ACC_sd_base = float(np.nanmedian(lower))
    except Exception:
        ACC_sd_base = np.nan

    # GYRO_base: compute per-sample energy, apply light smoothing, then take median
    if gyro_series is not None:
        try:
            gyro = np.asarray(gyro_series)
            if gyro.ndim == 1:
                gyro_energy = gyro ** 2
            else:
                gyro_energy = np.sum(gyro ** 2, axis=1).astype(float)

            # smoothing window (relative); if insufficient samples, skip smoothing
            if gyro_energy.size >= 5:
                smooth_w = max(1, int(gyro_energy.size // 50))
                if smooth_w > 1:
                    # simple moving average
                    cumsum = np.cumsum(np.insert(gyro_energy, 0, 0.0))
                    smoothed = (cumsum[smooth_w:] - cumsum[:-smooth_w]) / float(smooth_w)
                    gyro_energy_used = smoothed
                else:
                    gyro_energy_used = gyro_energy
            else:
                gyro_energy_used = gyro_energy

            GYRO_base = float(_safe_median(gyro_energy_used))
        except Exception:
            GYRO_base = np.nan
    else:
        GYRO_base = np.nan

    return Baseline(HR_base=HR_base,
                    RMSSD_base=RMSSD_base,
                    PAV_base=PAV_base,
                    TEMP_base=TEMP_base,
                    ACC_sd_base=ACC_sd_base,
                    GYRO_base=GYRO_base)


def compute_motion_features(acc_window: np.ndarray, gyro_window: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute motion-related features for an 8-second window.

    Args:
        acc_window: array-like with shape (N,3) or (N,) for single-axis.
        gyro_window: optional array-like with shape (N,3) or (N,)

    Returns:
        Dict containing:
            - acc_mag: numpy array of magnitudes
            - acc_sd: standard deviation of acc_mag
            - gyro_energy: sum-square energy over gyro samples (mean energy)
            - state: 'REST' or 'ACTIVE' computed relative to baseline in classify step
    """
    acc = np.asarray(acc_window)
    if acc.ndim == 1:
        acc_mag = acc
    else:
        # Avoid overflow by converting to float
        acc = acc.astype(float)
        acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

    acc_sd = float(np.nanstd(acc_mag))

    gyro_energy = np.nan
    if gyro_window is not None:
        g = np.asarray(gyro_window)
        if g.size > 0:
            if g.ndim == 1:
                gyro_energy = float(np.nanmean(g ** 2))
            else:
                gyro_energy = float(np.nanmean(np.sum(g ** 2, axis=1)))

    return {
        "acc_mag": acc_mag,
        "acc_sd": acc_sd,
        "gyro_energy": gyro_energy,
    }


def compute_hr_features(HR_window: np.ndarray, baseline: Baseline) -> Dict[str, float]:
    """
    Compute HR mean and HR_delta relative to baseline.

    HR_window: array-like heart rate samples (bpm) in the 8s window.
    baseline: Baseline object with HR_base
    """
    hr_arr = np.asarray(HR_window)
    HR_mean = float(np.nanmean(hr_arr)) if hr_arr.size > 0 else np.nan
    HR_delta = HR_mean - (baseline.HR_base if baseline.HR_base is not None else np.nan)
    return {"HR_mean": HR_mean, "HR_delta": HR_delta}


def compute_hrv_features(IBI_window: np.ndarray, baseline: Baseline) -> Dict[str, float]:
    """
    Compute RMSSD and HRV_ratio.

    IBI_window: array-like inter-beat intervals. Should be in ms or seconds consistently.
    """
    ibi_arr = np.asarray(IBI_window)
    RMSSD = _compute_rmssd(ibi_arr)
    # Avoid division by zero
    RMSSD_base = baseline.RMSSD_base if (baseline.RMSSD_base is not None and not np.isnan(baseline.RMSSD_base)) else EPS
    HRV_ratio = RMSSD / (RMSSD_base if abs(RMSSD_base) > EPS else EPS)
    return {"RMSSD": RMSSD, "HRV_ratio": HRV_ratio}


def compute_ppg_features(PPG_window: np.ndarray, baseline: Baseline) -> Dict[str, float]:
    """
    Compute PAV and PAV_ratio from raw PPG window.
    """
    ppg_arr = np.asarray(PPG_window)
    PAV = _compute_ppg_pav(ppg_arr)
    PAV_base = baseline.PAV_base if (baseline.PAV_base is not None and not np.isnan(baseline.PAV_base)) else EPS
    PAV_ratio = PAV / (PAV_base if abs(PAV_base) > EPS else EPS)
    return {"PAV": PAV, "PAV_ratio": PAV_ratio}


def compute_temperature_features(temp_window: np.ndarray, baseline: Baseline) -> Dict[str, float]:
    """
    Compute TEMP_mean and TEMP_delta.
    """
    t_arr = np.asarray(temp_window)
    TEMP_mean = float(np.nanmean(t_arr)) if t_arr.size > 0 else np.nan
    TEMP_delta = TEMP_mean - (baseline.TEMP_base if baseline.TEMP_base is not None else np.nan)
    return {"TEMP_mean": TEMP_mean, "TEMP_delta": TEMP_delta}


def classify_mood(features: Dict[str, Any], baseline: Baseline) -> str:
    """
    Apply the EXACT final conditional mood classification logic provided.

    features should contain at least:
      - 'state' (string): 'REST' or 'ACTIVE'
      - 'HR_delta'
      - 'HRV_ratio'
      - 'PAV_ratio'
      - 'TEMP_delta'
      - 'agitation_flag' (boolean)

    Returns one of the strings dictated by the logic. Per the exact rule: when not REST,
    returns the literal "ACTIVE / UNKNOWN".

    NOTE: This implements the specified thresholds exactly and in the specified order.
    """
    state = features.get("state", "ACTIVE")

    # Context check: only classify when in REST state. Treat MICRO_MOVEMENT same as ACTIVE
    if state != "REST":
        return "ACTIVE / UNKNOWN"

    HR_delta = features.get("HR_delta", np.nan)
    HRV_ratio = features.get("HRV_ratio", np.nan)
    PAV_ratio = features.get("PAV_ratio", np.nan)
    TEMP_delta = features.get("TEMP_delta", np.nan)
    TEMP_trend = features.get("TEMP_trend", np.nan)
    agitation_flag = bool(features.get("agitation_flag", False))

    # Stress (updated thresholds)
    # Use TEMP_trend (long-term) if available, otherwise fall back to short-term TEMP_delta
    temp_stress_ok = False
    if not np.isnan(TEMP_trend):
        temp_stress_ok = (TEMP_trend < -0.10)
    else:
        temp_stress_ok = (not np.isnan(TEMP_delta)) and (TEMP_delta < -0.15)

    if (HR_delta is not None and HR_delta > 10
            and HRV_ratio < 0.75
            and PAV_ratio < 0.90
            and temp_stress_ok):
        return "STRESS"

    # Calm (updated thresholds)
    if (HR_delta is not None and HR_delta < -4
            and HRV_ratio > 1.15
            and PAV_ratio > 1.05):
        return "CALM"

    # Agitated (updated agitation sensitivity)
    if (HR_delta is not None and HR_delta > 8
            and (0.85 <= HRV_ratio <= 1.10)
            and agitation_flag):
        return "AGITATED"

    # Neutral (updated ranges)
    if ((HR_delta is not None) and (-4 <= HR_delta <= 4)
            and (0.85 <= HRV_ratio <= 1.15)):
        return "NEUTRAL"

    # Default
    return "UNKNOWN"


def process_window(window_data: Dict[str, Any], baseline: Baseline, temp_long_window: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Process a single 8-second window and return classification plus computed features.

    window_data expected keys:
      - 'hr': 1D array-like of HR samples (bpm).
      - 'ibi': 1D array-like of IBI (ms or seconds, consistent with baseline)
      - 'ppg': 1D array-like raw PPG waveform
      - 'acc': Nx3 array-like accelerometer samples for the window
      - 'temp': 1D array-like of temperature samples
      - 'gyro': optional Nx3 array-like gyro samples

    Optional argument:
      - temp_long_window: array-like of temperature samples over a longer period
        (recommended ~3 minutes) to compute `TEMP_trend`.

    Returns a dict with:
      - 'classification': string result
      - 'features': dict of computed features
      - 'baseline': baseline.as_dict()
    """
    # Extract inputs with safe defaults
    hr_window = window_data.get("hr", np.array([]))
    ibi_window = window_data.get("ibi", np.array([]))
    ppg_window = window_data.get("ppg", np.array([]))
    acc_window = window_data.get("acc", np.array([]))
    temp_window = window_data.get("temp", np.array([]))
    gyro_window = window_data.get("gyro", None)

    # Motion features
    motion = compute_motion_features(acc_window, gyro_window)
    acc_sd = motion["acc_sd"]

    # Motion gating (three-stage): REST, MICRO_MOVEMENT, ACTIVE
    ACC_sd_base = baseline.ACC_sd_base if (baseline.ACC_sd_base is not None and not np.isnan(baseline.ACC_sd_base)) else EPS
    if acc_sd < (ACC_sd_base * 1.35):
        state = "REST"
    elif acc_sd < (ACC_sd_base * 2.25):
        state = "MICRO_MOVEMENT"
    else:
        state = "ACTIVE"

    # HR features
    hr_feats = compute_hr_features(hr_window, baseline)

    # HRV features
    hrv_feats = compute_hrv_features(ibi_window, baseline)

    # PPG features
    ppg_feats = compute_ppg_features(ppg_window, baseline)

    # Temperature: compute short-term temp and optional long-term trend
    temp_feats = compute_temperature_features(temp_window, baseline)
    TEMP_trend = np.nan
    if temp_long_window is not None:
        try:
            t_long = np.asarray(temp_long_window)
            if t_long.size >= 2:
                recent_mean = np.nanmean(np.asarray(temp_window)) if np.asarray(temp_window).size > 0 else np.nan
                long_mean = float(np.nanmean(t_long))
                TEMP_trend = recent_mean - long_mean
            else:
                TEMP_trend = np.nan
        except Exception:
            TEMP_trend = np.nan

    # Agitation flag (tuned thresholds)
    gyro_energy = motion.get("gyro_energy", np.nan)
    GYRO_base = baseline.GYRO_base if (baseline.GYRO_base is not None and not np.isnan(baseline.GYRO_base)) else np.nan
    agitation_flag = False
    try:
        agitation_flag = (acc_sd > (ACC_sd_base * 2.0)) or (
            (not np.isnan(gyro_energy)) and (not np.isnan(GYRO_base)) and (gyro_energy > (GYRO_base * 3.0)))
    except Exception:
        agitation_flag = (acc_sd > (ACC_sd_base * 2.0))

    # Compose features
    features = {
        "state": state,
        "acc_sd": acc_sd,
        "gyro_energy": gyro_energy,
        **hr_feats,
        **hrv_feats,
        **ppg_feats,
        **temp_feats,
        "TEMP_trend": TEMP_trend,
        "agitation_flag": agitation_flag,
    }

    classification = classify_mood(features, baseline)

    return {"classification": classification, "features": features, "baseline": baseline.as_dict()}


# Optional convenience: an offline sliding-window processor (not required but helpful)
def sliding_window_process(stream: Dict[str, np.ndarray], baseline: Baseline, window_samples: int, hop_samples: int) -> pd.DataFrame:
    """
    Process multi-channel streams with fixed-length sample windows. Returns a pandas DataFrame of results.

    Parameters:
        stream: dict with keys 'hr','ibi','ppg','acc','temp','gyro'. Each value is a numpy array.
        baseline: Baseline object
        window_samples: number of samples per window (for sample-aligned signals)
        hop_samples: hop length between windows

    Note: This function assumes all signals are aligned in sample counts. For a real device,
    you would need to resample or align by time.
    """
    # Determine length from one of the signals
    length = None
    for k in ["ppg", "hr", "ibi", "temp", "acc"]:
        if k in stream and stream[k] is not None:
            length = len(stream[k])
            break
    if length is None:
        raise ValueError("No usable signals in stream")

    results = []
    for start in range(0, length - window_samples + 1, hop_samples):
        window = {}
        for k in ["hr", "ibi", "ppg", "acc", "temp", "gyro"]:
            v = stream.get(k, None)
            if v is None:
                window[k] = np.array([])
            else:
                window[k] = np.asarray(v[start:start + window_samples])
        out = process_window(window, baseline)
        row = {"start": start, "classification": out["classification"]}
        # flatten some features
        feats = out["features"]
        row.update({
            "HR_mean": feats.get("HR_mean", np.nan),
            "HR_delta": feats.get("HR_delta", np.nan),
            "RMSSD": feats.get("RMSSD", np.nan),
            "HRV_ratio": feats.get("HRV_ratio", np.nan),
            "PAV": feats.get("PAV", np.nan),
            "PAV_ratio": feats.get("PAV_ratio", np.nan),
            "TEMP_delta": feats.get("TEMP_delta", np.nan),
            "acc_sd": feats.get("acc_sd", np.nan),
            "agitation_flag": feats.get("agitation_flag", False),
        })
        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Minimal usage example (offline simulation)
    print("This module provides functions for baseline computation and window processing.")
    print("Import and call compute_baselines(...) and process_window(...) in your firmware simulation.")
