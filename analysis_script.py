#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuM continuous current-clamp analysis pipeline (minimal version)
================================================================

Purpose
-------
Linear, easy-to-follow pipeline for Spyder / Anaconda use on SuM neurons.
For each .mat file, the script runs this sequence:

1. Load file
2. Detect current step
3. Make QC figure first
4. Compute passive properties: RMP, Rin, tau_hyper, tau_depol, Cm, sag
5. Detect spikes
6. Compute rheobase and absolute F-I
7. Compute normalized F-I to rheobase
8. Compute early vs late phase-plane
9. Print compact summary in console
10. Save minimal CSV outputs + 3 figures

Expected MATLAB file structure
------------------------------
inputData   -> membrane voltage (mV), shape [samples x sweeps]
outputData  -> injected current (pA), shape [samples x sweeps]
Pars.sampleRate -> sampling rate (Hz)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress, gaussian_kde
from sklearn.mixture import GaussianMixture

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

def matlab_field(obj, field: str, default=None):
    try:
        return getattr(obj, field)
    except Exception:
        try:
            return obj[field]
        except Exception:
            return default


def load_mat_file(path: Path) -> Dict[str, np.ndarray]:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "inputData" not in mat:
        raise KeyError(f"{path.name}: inputData not found")
    V = np.asarray(mat["inputData"], dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    I = None
    if "outputData" in mat:
        I = np.asarray(mat["outputData"], dtype=float)
        if I.ndim == 1:
            I = I[:, None]
    fs = None
    if "Pars" in mat:
        fs = matlab_field(mat["Pars"], "sampleRate", None)
    if fs is None and "sampleRate" in mat:
        fs = float(np.asarray(mat["sampleRate"]).squeeze())
    if fs is None:
        raise KeyError(f"{path.name}: sample rate not found")
    return {"V": V, "I": I, "fs": float(fs)}


def parse_cell_id_from_filename(path: Path) -> str:
    stem = path.stem
    import re
    m = re.match(r"(.+?_experiment[0-9]+)trial[0-9]+$", stem)
    if m:
        return m.group(1)
    return stem


def exp_rise(t, a, tau, c):
    return a * (1 - np.exp(-t / tau)) + c


def safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.size(x) else np.nan


def safe_sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def set_plot_style():
    if not PLOTTING_AVAILABLE:
        return
    plt.rcParams["font.family"] = CONFIG["font_family"]
    plt.rcParams["font.size"] = CONFIG["font_size"]


def detect_step_bounds(I: Optional[np.ndarray], fs: float, n_sweeps: int) -> Tuple[int, int, np.ndarray]:
    if I is None:
        fallback = CONFIG["fallback_current_steps_pA"]
        if fallback is None:
            raise ValueError("outputData missing and fallback_current_steps_pA not provided")
        current_levels = np.asarray(fallback, dtype=float)
        if current_levels.size != n_sweeps:
            raise ValueError("fallback_current_steps_pA length != number of sweeps")
        start = int(0.2 * fs)
        end = int(1.2 * fs)
        return start, end, current_levels
    probe = I[:, 0]
    baseline = np.median(probe[: max(10, int(0.1 * len(probe)))])
    idx = np.where(np.abs(probe - baseline) > 1e-6)[0]
    if idx.size == 0:
        raise ValueError("No current step detected in outputData")
    start = int(idx[0])
    end = int(idx[-1]) + 1
    pad = max(1, int(0.1 * (end - start)))
    current_levels = []
    for sw in range(I.shape[1]):
        current = np.median(I[start + pad:end - pad, sw])
        current_levels.append(float(current))
    return start, end, np.asarray(current_levels, dtype=float)


def get_windows(start: int, end: int, fs: float) -> Dict[str, slice]:
    base_n = int(CONFIG["baseline_pre_step_s"] * fs)
    baseline = slice(max(0, start - base_n), start)
    steady_n = int(CONFIG["steady_state_window_s"] * fs)
    steady = slice(max(start, end - steady_n), end)
    return {"baseline": baseline, "steady": steady, "step": slice(start, end)}


def detect_spikes_in_sweep(v: np.ndarray, fs: float, step_start: int, step_end: int) -> List[Tuple[int, int]]:
    dt = 1.0 / fs
    dvdt = np.gradient(v, dt) / 1000.0
    refractory_pts = max(1, int(CONFIG["refractory_ms"] * fs / 1000.0))
    peaks, _ = find_peaks(v[step_start:step_end], height=CONFIG["peak_min_mV"], distance=refractory_pts)
    peaks = peaks + step_start
    spikes = []
    last_peak = -10 ** 9
    search_back_pts = int(0.005 * fs)
    for pk in peaks:
        if pk - last_peak < refractory_pts:
            continue
        search_start = max(step_start, pk - search_back_pts)
        crossings = np.where(dvdt[search_start:pk] >= CONFIG["dvdt_threshold_mV_per_ms"])[0]
        if crossings.size == 0:
            continue
        thr = search_start + int(crossings[0])
        amplitude = v[pk] - v[thr]
        if amplitude < CONFIG["amplitude_min_mV"]:
            continue
        spikes.append((thr, pk))
        last_peak = pk
    return spikes


def extract_waveform_segment(v: np.ndarray, fs: float, threshold_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / fs
    pre_pts = int(CONFIG["pre_spike_ms"] * fs / 1000.0)
    post_pts = int(CONFIG["post_spike_ms"] * fs / 1000.0)
    w0 = max(0, threshold_idx - pre_pts)
    w1 = min(len(v), threshold_idx + post_pts)
    seg_v = v[w0:w1].copy()
    seg_dvdt = np.gradient(seg_v, dt) / 1000.0
    t_rel_ms = (np.arange(w0, w1) - threshold_idx) / fs * 1000.0
    return t_rel_ms, seg_v, seg_dvdt


def compute_passive_metrics(V: np.ndarray, current_levels: np.ndarray, fs: float, start: int, end: int) -> pd.DataFrame:
    windows = get_windows(start, end, fs)
    rows = []
    for sw in range(V.shape[1]):
        v = V[:, sw]
        baseline = safe_mean(v[windows["baseline"]])
        steady = safe_mean(v[windows["steady"]])
        deltaV = steady - baseline
        step_trace = v[windows["step"]]
        min_v = float(np.nanmin(step_trace))
        sag = steady - min_v
        has_spike = len(detect_spikes_in_sweep(v, fs, start, end)) > 0
        rows.append({
            "sweep_index": sw,
            "current_pA": float(current_levels[sw]),
            "baseline_mV": baseline,
            "steady_mV": steady,
            "deltaV_mV": deltaV,
            "min_mV": min_v,
            "sag_mV": sag,
            "has_spike": has_spike,
        })
    return pd.DataFrame(rows)


def fit_tau_hyper(V: np.ndarray, sweep_idx: int, baseline_mV: float, fs: float, start: int, end: int) -> Tuple[
    float, float]:
    fit_len = int(CONFIG["tau_hyper_fit_window_s"] * fs)
    fit_end = min(end, start + fit_len)
    t = np.arange(fit_end - start) / fs
    trace = V[start:fit_end, sweep_idx]
    y = np.abs(trace - baseline_mV)
    if len(y) < 10 or np.nanmax(y) <= 0:
        return np.nan, np.nan
    try:
        p0 = [float(np.nanmax(y)), 0.020, float(y[0])]
        bounds = ([0, 1e-4, -np.inf], [np.inf, 1.0, np.inf])
        popt, _ = curve_fit(exp_rise, t, y, p0=p0, bounds=bounds, maxfev=10000)
        fit = exp_rise(t, *popt)
        ss_res = np.nansum((y - fit) ** 2)
        ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(popt[1] * 1000.0), float(r2)
    except Exception:
        return np.nan, np.nan


def fit_tau_depol(V: np.ndarray, passive_df: pd.DataFrame, fs: float, start: int, end: int) -> Tuple[
    float, float, Optional[int]]:
    candidates = passive_df[(passive_df["current_pA"] > 0) & (~passive_df["has_spike"])].sort_values("current_pA")
    if candidates.empty:
        return np.nan, np.nan, None
    row = candidates.iloc[-1]
    sw = int(row["sweep_index"])
    baseline_mV = float(row["baseline_mV"])
    fit_len = int(CONFIG["tau_depol_fit_window_s"] * fs)
    fit_end = min(end, start + fit_len)
    t = np.arange(fit_end - start) / fs
    trace = V[start:fit_end, sw]
    y = trace - baseline_mV
    if len(y) < 10 or np.nanmax(y) <= 0:
        return np.nan, np.nan, sw
    try:
        p0 = [float(np.nanmax(y)), 0.010, float(y[0])]
        bounds = ([0, 1e-4, -np.inf], [np.inf, 1.0, np.inf])
        popt, _ = curve_fit(exp_rise, t, y, p0=p0, bounds=bounds, maxfev=10000)
        fit = exp_rise(t, *popt)
        ss_res = np.nansum((y - fit) ** 2)
        ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(popt[1] * 1000.0), float(r2), sw
    except Exception:
        return np.nan, np.nan, sw


def compute_active_metrics(V: np.ndarray, current_levels: np.ndarray, fs: float, start: int, end: int) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    dt = 1.0 / fs
    spike_rows = []
    fi_rows = []
    for sw in range(V.shape[1]):
        v = V[:, sw]
        dvdt = np.gradient(v, dt) / 1000.0
        spikes = detect_spikes_in_sweep(v, fs, start, end)
        duration_s = (end - start) / fs
        n_spikes = len(spikes)
        firing_rate_hz = n_spikes / duration_s
        if n_spikes >= 2:
            peak_times = np.array([pk for _, pk in spikes], dtype=float) / fs
            isi_ms = np.diff(peak_times) * 1000.0
            mean_isi_ms = float(np.nanmean(isi_ms))
            cv_isi = float(np.nanstd(isi_ms, ddof=1) / np.nanmean(isi_ms)) if isi_ms.size >= 2 else np.nan
            first_inst_freq = 1000.0 / isi_ms[0] if isi_ms[0] > 0 else np.nan
        else:
            mean_isi_ms = np.nan
            cv_isi = np.nan
            first_inst_freq = np.nan
        fi_rows.append({
            "sweep_index": sw,
            "current_pA": float(current_levels[sw]),
            "n_spikes": n_spikes,
            "firing_rate_Hz": firing_rate_hz,
            "first_inst_freq_Hz": first_inst_freq,
            "mean_ISI_ms": mean_isi_ms,
            "CV_ISI": cv_isi,
        })
        for i_spk, (thr, pk) in enumerate(spikes, start=1):
            threshold_v = float(v[thr])
            peak_v = float(v[pk])
            amplitude = peak_v - threshold_v
            pre_pts = int(CONFIG["pre_spike_ms"] * fs / 1000.0)
            post_pts = int(CONFIG["post_spike_ms"] * fs / 1000.0)
            w0 = max(0, thr - pre_pts)
            w1 = min(len(v), pk + post_pts)
            seg_dvdt = dvdt[w0:w1]
            upstroke = float(np.nanmax(seg_dvdt))
            downstroke = float(np.nanmin(seg_dvdt))
            half_amp = threshold_v + amplitude / 2.0
            left_candidates = np.where(v[thr:pk + 1] >= half_amp)[0]
            if left_candidates.size:
                left_idx = thr + int(left_candidates[0])
                right_candidates = np.where(v[pk:w1] <= half_amp)[0]
                half_width_ms = (pk + int(
                    right_candidates[0]) - left_idx) / fs * 1000.0 if right_candidates.size else np.nan
            else:
                half_width_ms = np.nan
            ahp_end = min(len(v), pk + int(0.020 * fs))
            if ahp_end > pk + 1:
                post_peak = v[pk:ahp_end]
                ahp_rel_idx = int(np.nanargmin(post_peak))
                ahp_idx = pk + ahp_rel_idx
                fast_ahp_mV = float(v[ahp_idx] - threshold_v)
                fast_ahp_time_ms = (ahp_idx - pk) / fs * 1000.0
            else:
                fast_ahp_mV = np.nan
                fast_ahp_time_ms = np.nan
            spike_rows.append({
                "sweep_index": sw,
                "spike_number": i_spk,
                "current_pA": float(current_levels[sw]),
                "threshold_time_s": thr / fs,
                "peak_time_s": pk / fs,
                "threshold_mV": threshold_v,
                "peak_mV": peak_v,
                "overshoot_mV": peak_v,
                "amplitude_mV": amplitude,
                "half_width_ms": half_width_ms,
                "upstroke_mV_per_ms": upstroke,
                "downstroke_mV_per_ms": downstroke,
                "fast_ahp_mV": fast_ahp_mV,
                "fast_ahp_time_ms": fast_ahp_time_ms,
            })
    return pd.DataFrame(spike_rows), pd.DataFrame(fi_rows)


def compute_fi_summary(fi_df: pd.DataFrame) -> Dict[str, float]:
    pos = fi_df[(fi_df["current_pA"] > 0) & (fi_df["n_spikes"] > 0)].sort_values("current_pA")
    if pos.empty:
        return {
            "rheobase_pA": np.nan,
            "fi_slope_Hz_per_pA": np.nan,
            "fi_intercept_Hz": np.nan,
            "fi_r2": np.nan,
            "max_firing_rate_Hz": 0.0,
        }
    rheobase = float(pos["current_pA"].min())
    fit_df = pos[pos["current_pA"] <= rheobase + CONFIG["fi_fit_max_delta_pA"]].copy()
    if fit_df.shape[0] >= 2:
        # usa esplicitamente I normalizzata
        fit_df = fit_df.copy()
        fit_df["I_norm_pA"] = fit_df["current_pA"] - rheobase

        x = fit_df["I_norm_pA"].to_numpy(dtype=float)
        y = fit_df["firing_rate_Hz"].to_numpy(dtype=float)

        # opzionale: limita a range standard
        # fit_df = fit_df[fit_df["I_norm_pA"] <= CONFIG["fi_fit_max_delta_pA"]]

        if np.unique(x).size >= 2:
            lr = linregress(x, y)
            slope, intercept, r2 = float(lr.slope), float(lr.intercept), float(lr.rvalue ** 2)
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan
    else:
        slope, intercept, r2 = np.nan, np.nan, np.nan
    return {
        "rheobase_pA": rheobase,
        "fi_slope_Hz_per_pA": slope,
        "fi_intercept_Hz": intercept,
        "fi_r2": r2,
        "max_firing_rate_Hz": float(fi_df["firing_rate_Hz"].max()),
    }


def add_normalized_fi(fi_df: pd.DataFrame, rheobase_pA: float) -> pd.DataFrame:
    fi_df = fi_df.copy()
    if np.isfinite(rheobase_pA):
        fi_df["I_norm_pA"] = fi_df["current_pA"] - rheobase_pA
        step = CONFIG["normalized_bin_step_pA"]
        fi_df["I_norm_bin_pA"] = np.round(fi_df["I_norm_pA"] / step) * step
    else:
        fi_df["I_norm_pA"] = np.nan
        fi_df["I_norm_bin_pA"] = np.nan
    return fi_df


def compute_early_late_phaseplane(V: np.ndarray, fs: float, start: int, end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    early_numbers = set(CONFIG["early_spike_numbers"])
    late_numbers = set(CONFIG["late_spike_numbers"])
    early_segments = []
    late_segments = []
    for sw in range(V.shape[1]):
        v = V[:, sw]
        spikes = detect_spikes_in_sweep(v, fs, start, end)
        for i_spk, (thr, _pk) in enumerate(spikes, start=1):
            t_rel_ms, seg_v, seg_dvdt = extract_waveform_segment(v, fs, thr)
            if i_spk in early_numbers:
                early_segments.append((t_rel_ms, seg_v, seg_dvdt))
            elif i_spk in late_numbers:
                late_segments.append((t_rel_ms, seg_v, seg_dvdt))
    phase_rows = []
    waveform_rows = []

    def average_segments(segments, label):
        if len(segments) == 0:
            return
        min_len = min(len(seg[1]) for seg in segments)
        t_rel = segments[0][0][:min_len]
        Vmat = np.vstack([seg[1][:min_len] for seg in segments])
        Dmat = np.vstack([seg[2][:min_len] for seg in segments])
        mean_v = np.nanmean(Vmat, axis=0)
        mean_dvdt = np.nanmean(Dmat, axis=0)
        thr_idx = int(np.argmin(np.abs(t_rel)))
        peak_idx = int(np.nanargmax(mean_v))
        phase_rows.append({
            "label": label,
            "n_spikes_averaged": len(segments),
            "threshold_mV": float(mean_v[thr_idx]),
            "peak_mV": float(mean_v[peak_idx]),
            "upstroke_mV_per_ms": float(np.nanmax(mean_dvdt)),
            "downstroke_mV_per_ms": float(np.nanmin(mean_dvdt)),
        })
        for i in range(min_len):
            waveform_rows.append({
                "label": label,
                "time_rel_ms": float(t_rel[i]),
                "V_mV": float(mean_v[i]),
                "dVdt_mV_per_ms": float(mean_dvdt[i]),
            })

    average_segments(early_segments, "early")
    average_segments(late_segments, "late")
    return pd.DataFrame(phase_rows), pd.DataFrame(waveform_rows)


def assign_group_from_slope(slope: float) -> str:
    if not np.isfinite(slope):
        return "Unclassified"
    return "LG" if slope < CONFIG["lg_hg_slope_threshold_Hz_per_pA"] else "HG"

def summarize_cells_from_trials(batch_df: pd.DataFrame) -> pd.DataFrame:
    if batch_df.empty:
        return pd.DataFrame()
    batch_df = batch_df.copy()
    #batch_df["cell_id"] = batch_df["file"].apply(lambda x: parse_cell_id_from_filename(Path(x)))
    numeric_cols = [
        "RMP_mV",
        "Rin_linear_MOhm",
        "Rin_linear_r2",
        "Rin_strongest_hyper_MOhm",
        "tau_hyper_ms",
        "tau_hyper_r2",
        "tau_depol_ms",
        "tau_depol_r2",
        "Cm_from_tau_hyper_pF",
        "Cm_from_tau_depol_pF",
        "sag_mV_strongest_hyper",
        "rheobase_pA",
        "fi_slope_Hz_per_pA",
        "fi_r2",
        "max_firing_rate_Hz",
        "first_spike_threshold_mV",
        "first_spike_peak_mV",
        "first_spike_half_width_ms",
        "first_spike_upstroke_mV_per_ms",
        "early_upstroke_mV_per_ms",
        "late_upstroke_mV_per_ms",
        "late_vs_early_upstroke_change_pct",
    ]
    agg = {col: "mean" for col in numeric_cols if col in batch_df.columns}
    agg["file"] = "count"
    cell_df = batch_df.groupby("cell_id", as_index=False).agg(agg)
    cell_df = cell_df.rename(columns={"file": "n_trials"})
    return cell_df


def classify_cells_from_slope_distribution(cell_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if cell_df.empty or "fi_slope_Hz_per_pA" not in cell_df.columns:
        return cell_df, np.nan
    out = cell_df.copy()
    valid = out["fi_slope_Hz_per_pA"].to_numpy(dtype=float)
    mask = np.isfinite(valid) & (valid > 0)
    if np.sum(mask) < 4:
        out["group"] = "Unclassified"
        out["log10_slope"] = np.nan
        return out, np.nan
    log_slope = np.full_like(valid, np.nan, dtype=float)
    log_slope[mask] = np.log10(valid[mask])
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(log_slope[mask].reshape(-1, 1))
    pred = gmm.predict(log_slope[mask].reshape(-1, 1))
    means = gmm.means_.flatten()
    lg_component = int(np.argmin(means))
    hg_component = int(np.argmax(means))
    labels = np.full(len(out), "Unclassified", dtype=object)
    labels_idx = np.where(mask)[0]
    for idx_local, comp in zip(labels_idx, pred):
        labels[idx_local] = "LG" if comp == lg_component else "HG"
    out["log10_slope"] = log_slope
    out["group"] = labels
    xgrid = np.linspace(np.nanmin(log_slope[mask]) - 0.5, np.nanmax(log_slope[mask]) + 0.5, 4000)
    probs = gmm.predict_proba(xgrid.reshape(-1, 1))
    diff = np.abs(probs[:, lg_component] - probs[:, hg_component])
    boundary_log = xgrid[np.argmin(diff)]
    boundary_linear = 10 ** boundary_log
    return out, float(boundary_linear)


def summarize_groups_from_cells(cell_df: pd.DataFrame) -> pd.DataFrame:
    if cell_df.empty:
        return pd.DataFrame()
    rows = []
    numeric_cols = [
        "RMP_mV",
        "Rin_linear_MOhm",
        "Rin_strongest_hyper_MOhm",
        "tau_hyper_ms",
        "tau_depol_ms",
        "Cm_from_tau_hyper_pF",
        "Cm_from_tau_depol_pF",
        "sag_mV_strongest_hyper",
        "rheobase_pA",
        "fi_slope_Hz_per_pA",
        "max_firing_rate_Hz",
        "first_spike_threshold_mV",
        "first_spike_peak_mV",
        "first_spike_half_width_ms",
        "first_spike_upstroke_mV_per_ms",
        "early_upstroke_mV_per_ms",
        "late_upstroke_mV_per_ms",
        "late_vs_early_upstroke_change_pct",
    ]
    for grp, sub in cell_df.groupby("group"):
        row = {"group": grp, "n_cells": len(sub)}
        for col in numeric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = float(np.nanmean(sub[col]))
                row[f"{col}_sem"] = safe_sem(sub[col])
        rows.append(row)
    return pd.DataFrame(rows)


def save_gain_distribution_figure(cell_df: pd.DataFrame, threshold_hz_per_pA: float, out_png: Path):
    if not PLOTTING_AVAILABLE or cell_df.empty:
        return
    valid = cell_df[np.isfinite(cell_df["fi_slope_Hz_per_pA"]) & (cell_df["fi_slope_Hz_per_pA"] > 0)].copy()
    if valid.empty:
        return
    set_plot_style()
    vals = np.log10(valid["fi_slope_Hz_per_pA"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = min(10, max(5, len(vals) // 2))
    ax.hist(vals, bins=bins, density=True, alpha=0.35, edgecolor="black")
    if len(vals) >= 3:
        kde = gaussian_kde(vals)
        xx = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 400)
        ax.plot(xx, kde(xx), linewidth=2)
    if np.isfinite(threshold_hz_per_pA) and threshold_hz_per_pA > 0:
        ax.axvline(np.log10(threshold_hz_per_pA), linestyle="--", linewidth=1.5)
    ax.set_xlabel("log10 F-I slope (Hz/pA)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of mean cell gain")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def build_group_phaseplane_dataset(root_out_dir: Path, cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, crow in cell_df.iterrows():
        cell_id = str(crow["cell_id"])
        group = str(crow["group"])
        trial_dirs = sorted(root_out_dir.glob(f"{cell_id}trial*"))
        for tdir in trial_dirs:
            phase_file = tdir / "phaseplane_features.csv"
            if not phase_file.exists():
                continue
            df = pd.read_csv(phase_file)
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "cell_id": cell_id,
                    "group": group,
                    "label": r.get("label", np.nan),
                    "n_spikes_averaged": r.get("n_spikes_averaged", np.nan),
                    "threshold_mV": r.get("threshold_mV", np.nan),
                    "peak_mV": r.get("peak_mV", np.nan),
                    "upstroke_mV_per_ms": r.get("upstroke_mV_per_ms", np.nan),
                    "downstroke_mV_per_ms": r.get("downstroke_mV_per_ms", np.nan),
                })
    return pd.DataFrame(rows)


def summarize_group_phaseplane(group_phase_df: pd.DataFrame) -> pd.DataFrame:
    if group_phase_df.empty:
        return pd.DataFrame()
    rows = []
    metrics = ["threshold_mV", "peak_mV", "upstroke_mV_per_ms", "downstroke_mV_per_ms"]
    for (grp, label), sub in group_phase_df.groupby(["group", "label"]):
        row = {"group": grp, "label": label, "n_rows": len(sub)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(np.nanmean(sub[metric]))
            row[f"{metric}_sem"] = safe_sem(sub[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def save_qc_figure(path: Path, V: np.ndarray, I: Optional[np.ndarray], fs: float, out_png: Path):
    if not PLOTTING_AVAILABLE:
        return
    set_plot_style()
    t = np.arange(V.shape[0]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for sw in range(V.shape[1]):
        axes[0].plot(t, V[:, sw], linewidth=0.8)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(path.name)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    if I is not None:
        for sw in range(I.shape[1]):
            axes[1].plot(t, I[:, sw], linewidth=0.8)
    axes[1].set_ylabel("Current (pA)")
    axes[1].set_xlabel("Time (s)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def save_fi_figures(fi_df: pd.DataFrame, fi_norm_df: pd.DataFrame, out_abs: Path, out_norm: Path):
    if not PLOTTING_AVAILABLE:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    tmp = fi_df.sort_values("current_pA")
    ax.scatter(tmp["current_pA"], tmp["firing_rate_Hz"], s=35)
    ax.set_xlabel("Injected current (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_abs, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    tmp = fi_norm_df.sort_values("I_norm_pA")
    ax.scatter(tmp["I_norm_pA"], tmp["firing_rate_Hz"], s=35)
    ax.set_xlabel("Injected current relative to rheobase (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_norm, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def save_phaseplane_figure(phase_waveform_df: pd.DataFrame, out_png: Path):
    if not PLOTTING_AVAILABLE or phase_waveform_df.empty:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in ["early", "late"]:
        sub = phase_waveform_df[phase_waveform_df["label"] == label]
        if not sub.empty:
            ax.plot(sub["V_mV"], sub["dVdt_mV_per_ms"], linewidth=1.8, label=label)
    ax.set_xlabel("Membrane voltage (mV)")
    ax.set_ylabel("dV/dt (mV/ms)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def analyze_one_file(path: Path, out_dir: Path):
    print("=" * 70)
    print(f"Analyzing: {path.name}")
    data = load_mat_file(path)
    V, I, fs = data["V"], data["I"], data["fs"]
    start, end, current_levels = detect_step_bounds(I, fs, V.shape[1])

    # 1. QC first
    save_qc_figure(path, V, I, fs, out_dir / "QC_voltage_current.png")

    # 2. Passive properties
    passive_df = compute_passive_metrics(V, current_levels, fs, start, end)
    zero_candidates = passive_df[np.isclose(passive_df["current_pA"], 0.0)]
    rmp = float(zero_candidates["baseline_mV"].mean()) if not zero_candidates.empty else float(passive_df["baseline_mV"].mean())
    hyper = passive_df[(passive_df["current_pA"] < 0) & (~passive_df["has_spike"])].sort_values("current_pA")
    if hyper.shape[0] >= 2:
        lr = linregress(hyper["current_pA"], hyper["deltaV_mV"])
        rin_linear = float(lr.slope * 1000.0)
        rin_r2 = float(lr.rvalue ** 2)
    else:
        rin_linear = np.nan
        rin_r2 = np.nan
    if not hyper.empty:
        strongest = hyper.iloc[0]
        strongest_I = float(strongest["current_pA"])
        strongest_deltaV = float(strongest["deltaV_mV"])
        rin_single = abs(strongest_deltaV / strongest_I) * 1000.0 if strongest_I != 0 else np.nan
        tau_h_ms, tau_h_r2 = fit_tau_hyper(V, int(strongest["sweep_index"]), float(strongest["baseline_mV"]), fs, start, end)
        sag_mV = float(strongest["sag_mV"])
    else:
        rin_single = np.nan
        tau_h_ms = np.nan
        tau_h_r2 = np.nan
        sag_mV = np.nan
    tau_d_ms, tau_d_r2, _ = fit_tau_depol(V, passive_df, fs, start, end)
    cm_hyper_pF = (tau_h_ms * 1000.0 / rin_linear) if np.isfinite(tau_h_ms) and np.isfinite(rin_linear) and rin_linear != 0 else np.nan
    cm_depol_pF = (tau_d_ms * 1000.0 / rin_linear) if np.isfinite(tau_d_ms) and np.isfinite(rin_linear) and rin_linear != 0 else np.nan

    # 3. Active properties and F-I
    spike_df, fi_df = compute_active_metrics(V, current_levels, fs, start, end)
    fi_summary = compute_fi_summary(fi_df)
    fi_norm_df = add_normalized_fi(fi_df, fi_summary["rheobase_pA"])
    save_fi_figures(fi_df, fi_norm_df, out_dir / "FI_absolute.png", out_dir / "FI_normalized.png")

    # 4. Early vs late phase-plane
    phase_feature_df, phase_waveform_df = compute_early_late_phaseplane(V, fs, start, end)
    save_phaseplane_figure(phase_waveform_df, out_dir / "PhasePlane_early_vs_late.png")
    early_row = phase_feature_df[phase_feature_df["label"] == "early"]
    late_row = phase_feature_df[phase_feature_df["label"] == "late"]
    if not early_row.empty and not late_row.empty:
        early_upstroke = float(early_row.iloc[0]["upstroke_mV_per_ms"])
        late_upstroke = float(late_row.iloc[0]["upstroke_mV_per_ms"])
        delta_upstroke_pct = ((late_upstroke - early_upstroke) / early_upstroke * 100.0) if early_upstroke != 0 else np.nan
    else:
        early_upstroke = np.nan
        late_upstroke = np.nan
        delta_upstroke_pct = np.nan

    # 5. Summary + save minimal outputs
    first_spikes = spike_df[spike_df["spike_number"] == 1]
    summary = pd.DataFrame([{
        "file": path.name,
        "cell_id": parse_cell_id_from_filename(path),
        "n_sweeps": int(V.shape[1]),
        "sample_rate_Hz": fs,
        "step_start_s": start / fs,
        "step_end_s": end / fs,
        "step_duration_s": (end - start) / fs,
        "current_min_pA": float(np.nanmin(current_levels)),
        "current_max_pA": float(np.nanmax(current_levels)),
        "RMP_mV": rmp,
        "Rin_linear_MOhm": rin_linear,
        "Rin_linear_r2": rin_r2,
        "Rin_strongest_hyper_MOhm": rin_single,
        "tau_hyper_ms": tau_h_ms,
        "tau_hyper_r2": tau_h_r2,
        "tau_depol_ms": tau_d_ms,
        "tau_depol_r2": tau_d_r2,
        "Cm_from_tau_hyper_pF": cm_hyper_pF,
        "Cm_from_tau_depol_pF": cm_depol_pF,
        "sag_mV_strongest_hyper": sag_mV,
        "rheobase_pA": fi_summary["rheobase_pA"],
        "fi_slope_Hz_per_pA": fi_summary["fi_slope_Hz_per_pA"],
        "fi_r2": fi_summary["fi_r2"],
        "max_firing_rate_Hz": fi_summary["max_firing_rate_Hz"],
        "group": assign_group_from_slope(fi_summary["fi_slope_Hz_per_pA"]),
        "first_spike_threshold_mV": float(first_spikes["threshold_mV"].mean()) if not first_spikes.empty else np.nan,
        "first_spike_peak_mV": float(first_spikes["peak_mV"].mean()) if not first_spikes.empty else np.nan,
        "first_spike_half_width_ms": float(first_spikes["half_width_ms"].mean()) if not first_spikes.empty else np.nan,
        "first_spike_upstroke_mV_per_ms": float(first_spikes["upstroke_mV_per_ms"].mean()) if not first_spikes.empty else np.nan,
        "early_upstroke_mV_per_ms": early_upstroke,
        "late_upstroke_mV_per_ms": late_upstroke,
        "late_vs_early_upstroke_change_pct": delta_upstroke_pct,
    }])
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    fi_df.to_csv(out_dir / "fi_curve.csv", index=False)
    fi_norm_df.to_csv(out_dir / "fi_normalized.csv", index=False)
    phase_feature_df.to_csv(out_dir / "phaseplane_features.csv", index=False)

    # 6. Print compact console report
    row = summary.iloc[0]
    print(f"RMP: {row['RMP_mV']:.2f} mV")
    print(f"Rin linear: {row['Rin_linear_MOhm']:.2f} MOhm")
    print(f"Tau hyper: {row['tau_hyper_ms']:.2f} ms")
    print(f"Tau depol: {row['tau_depol_ms']:.2f} ms")
    print(f"Cm (hyper): {row['Cm_from_tau_hyper_pF']:.2f} pF")
    print(f"Rheobase: {row['rheobase_pA']:.2f} pA")
    print(f"F-I slope: {row['fi_slope_Hz_per_pA']:.4f} Hz/pA")
    print(f"Group: {row['group']}")
    print(f"Early upstroke: {row['early_upstroke_mV_per_ms']:.2f} mV/ms")
    print(f"Late upstroke: {row['late_upstroke_mV_per_ms']:.2f} mV/ms")
    print(f"Late vs early change: {row['late_vs_early_upstroke_change_pct']:.2f} %")
    print(f"Saved in: {out_dir}")
    return summary

CONFIG = {
    "baseline_pre_step_s": 0.050,
    "steady_state_window_s": 0.050,
    "tau_hyper_fit_window_s": 0.150,
    "tau_depol_fit_window_s": 0.030,
    "dvdt_threshold_mV_per_ms": 10.0,
    "peak_min_mV": 0.0,
    "amplitude_min_mV": 40.0,
    "refractory_ms": 2.0,
    "pre_spike_ms": 5.0,
    "post_spike_ms": 12.0,
    "fi_fit_max_delta_pA": 40.0,
    "normalized_bin_step_pA": 10.0,
    "early_spike_numbers": [1, 2],
    "late_spike_numbers": [3, 4],
    "grouping_mode": "manual_threshold",
    "lg_hg_slope_threshold_Hz_per_pA": 0.08,
    "dpi": 400,
    "font_family": "Arial",
    "font_size": 12,
    "fallback_current_steps_pA": None,
    "output_directory":  "SuM_analysis_results_continuous_minimal_v4"
}


def analyze_files(file_paths: List[Path], root_out_dir: Path):
    root_out_dir.mkdir(parents=True, exist_ok=True)
    batch_rows = []
    qc_log = []
    for path in file_paths:
        file_out = root_out_dir / path.stem
        file_out.mkdir(exist_ok=True)
        try:
            summary = analyze_one_file(path, file_out)
            batch_rows.append(summary.iloc[0].to_dict())
            qc_log.append(f"[OK] {path.name}")
        except Exception as e:
            qc_log.append(f"[FAIL] {path.name}: {e}")
            print(f"FAILED: {path.name} -> {e}")
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(root_out_dir / "batch_summary_metrics.csv", index=False)
        cell_df = summarize_cells_from_trials(batch_df)
        cell_df, slope_threshold = classify_cells_from_slope_distribution(cell_df)
        cell_df.to_csv(root_out_dir / "cell_summary_metrics.csv", index=False)
        group_df = summarize_groups_from_cells(cell_df)
        group_df.to_csv(root_out_dir / "group_summary_from_cells.csv", index=False)
        save_gain_distribution_figure(cell_df, slope_threshold, root_out_dir / "Gain_distribution_cells.png")
        group_phase_df = build_group_phaseplane_dataset(root_out_dir, cell_df)
        group_phase_df.to_csv(root_out_dir / "group_phaseplane_trial_rows.csv", index=False)
        phase_summary_df = summarize_group_phaseplane(group_phase_df)
        phase_summary_df.to_csv(root_out_dir / "group_phaseplane_summary.csv", index=False)
        print(f"Distribution-based LG/HG threshold: {slope_threshold:.6f} Hz/pA")
    (root_out_dir / "qc_log.txt").write_text("\n".join(qc_log), encoding="utf-8")
    print("=" * 70)
    print(f"Batch analysis completed. Results saved in: {root_out_dir}")


def choose_files_gui() -> List[Path]:
    if not TK_AVAILABLE:
        raise RuntimeError("tkinter not available on this system")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select SuM .mat files", filetypes=[("MAT files", "*.mat")])
    root.update()
    root.destroy()
    return [Path(p) for p in file_paths]


def main():
    files = choose_files_gui() if TK_AVAILABLE else []
    if not files:
        print("No files selected.")
        return
    out_dir = files[0].parent / CONFIG["output_directory"]
    analyze_files(files, out_dir)


if __name__ == "__main__":
    main()
