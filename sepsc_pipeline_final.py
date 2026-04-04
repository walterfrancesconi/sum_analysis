import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks, medfilt, savgol_filter
from scipy.interpolate import interp1d


# ============================================================
# PARAMETRI FINALI CONCORDATI
# ============================================================

# Detection selettiva
DETECT_SIGMA = 1.6
PROM_FACTOR = 0.30
MIN_DISTANCE_MS = 2.0
AMP_MIN_PA = 3.5

# Vincoli cinetici
RISE_MIN_MS = 0.05
RISE_MAX_MS = 7.0
DECAY_MIN_MS = 0.1
DECAY_MAX_MS = 30.0
DURATION_MIN_MS = 0.3
DURATION_MAX_MS = 50.0

# Filtri
DETECTION_LOWPASS_HZ = 1500   # per detection
DISPLAY_LOWPASS_HZ = 300      # per il pannello lungo dei 30 s

# Baseline / event parsing
BASELINE_WINDOW_S = 0.08
PRE_ONSET_SEARCH_S = 0.01
LOCAL_BASELINE_S = 0.002
PEAK_REFINE_S = 0.0015
MAX_EVENT_LEN_S = 0.04
RETURN_LEVEL_FRACTION = 0.20

# Waveform media
AVG_PRE_MS = 12
AVG_POST_MS = 40
UPSAMPLE_FACTOR = 20

# Figura finale
REPRESENTATIVE_WINDOW_S = 30
FIRST_SECONDS_TO_PLOT = 20

# Classi di ampiezza concordate
AMP_BINS = [
    ("1", 3.5, 6.0),
    ("2", 6.0, 10.0),
    ("3", 10.0, np.inf),
]

# Files da analizzare
FILES = [
    "D:/2026/Apr/01/experiment002trial011.mat",
    #"experiment002trial011.mat",
    # aggiungi qui altri file se vuoi:
    # "experiment001trial012.mat",
    # "experiment001trial013.mat",
]

OUTPUT_DIR = "sEPSC_pipeline_output"


# ============================================================
# FUNZIONI BASE
# ============================================================

def load_trace(path):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    x = np.asarray(mat["inputData"]).squeeze().astype(float)
    fs = float(mat["Pars"].sampleRate)
    if np.isnan(x).any():
        x = x[~np.isnan(x)]
    return x, fs, os.path.basename(path)


def lowpass(sig, fs, cutoff):
    b, a = butter(2, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, sig)


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


# ============================================================
# DETECTION EVENTI sEPSC
# ============================================================

def detect_events(x, fs):
    x_det = lowpass(x, fs, DETECTION_LOWPASS_HZ)

    k = int(fs * BASELINE_WINDOW_S)
    if k % 2 == 0:
        k += 1

    baseline = medfilt(x_det, k)
    y = x_det - baseline
    sigma = robust_sigma(y)

    peaks, props = find_peaks(
        -y,
        height=DETECT_SIGMA * sigma,
        prominence=PROM_FACTOR * sigma,
        distance=int(MIN_DISTANCE_MS / 1000 * fs),
    )

    rows = []

    for pk_idx, pk in enumerate(peaks):
        amp0 = -y[pk]
        if amp0 < AMP_MIN_PA:
            continue

        # onset
        onset = pk
        pre_search = int(PRE_ONSET_SEARCH_S * fs)
        for i in range(pk, max(0, pk - pre_search), -1):
            if y[i] > -0.08 * amp0:
                onset = i
                break

        # refine peak
        refine = int(PEAK_REFINE_S * fs)
        p0 = max(0, pk - refine)
        p1 = min(len(x_det), pk + refine)
        if p1 <= p0:
            continue
        peak = p0 + np.argmin(x_det[p0:p1])

        # local baseline
        bl0 = max(0, onset - int(LOCAL_BASELINE_S * fs))
        bl1 = max(bl0 + 1, onset)
        local_baseline = np.median(x_det[bl0:bl1])

        amplitude = local_baseline - x_det[peak]
        if amplitude < AMP_MIN_PA:
            continue

        # rise 10–90
        lvl10 = local_baseline - 0.10 * amplitude
        lvl90 = local_baseline - 0.90 * amplitude
        idx10, idx90 = None, None
        for i in range(onset, peak):
            if idx10 is None and x_det[i] <= lvl10:
                idx10 = i
            if idx90 is None and x_det[i] <= lvl90:
                idx90 = i
                break

        rise_ms = np.nan
        if idx10 is not None and idx90 is not None and idx90 > idx10:
            rise_ms = (idx90 - idx10) / fs * 1000.0

        # end / duration
        max_len = int(MAX_EVENT_LEN_S * fs)
        end = min(len(x_det) - 1, peak + max_len)

        lvl_return = local_baseline - RETURN_LEVEL_FRACTION * amplitude
        for i in range(peak + 1, end):
            if x_det[i] >= lvl_return:
                end = i
                break

        duration_ms = (end - onset) / fs * 1000.0

        # decay to 37%
        lvl37 = local_baseline - 0.37 * amplitude
        decay_ms = np.nan
        for i in range(peak, end + 1):
            if x_det[i] >= lvl37:
                decay_ms = (i - peak) / fs * 1000.0
                break

        # charge
        cur = local_baseline - x_det[onset:end + 1]
        cur[cur < 0] = 0
        charge_pc = np.trapezoid(cur, dx=1 / fs) / 1000.0

        rows.append({
            "onset_idx": int(onset),
            "peak_idx": int(peak),
            "end_idx": int(end),
            "amp_pA": float(amplitude),
            "rise_ms": float(rise_ms),
            "decay_ms": float(decay_ms),
            "duration_ms": float(duration_ms),
            "charge_pC": float(charge_pc),
            "prominence": float(props["prominences"][pk_idx]),
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return x_det, df, sigma

    # filtri cinetici
    df = df[
        (df["amp_pA"] >= AMP_MIN_PA) &
        ((df["rise_ms"].between(RISE_MIN_MS, RISE_MAX_MS, inclusive="both")) | df["rise_ms"].isna()) &
        ((df["decay_ms"].between(DECAY_MIN_MS, DECAY_MAX_MS, inclusive="both")) | df["decay_ms"].isna()) &
        (df["duration_ms"].between(DURATION_MIN_MS, DURATION_MAX_MS, inclusive="both"))
    ].copy()

    # soppressione overlap
    df = df.sort_values(["peak_idx", "amp_pA"], ascending=[True, False]).reset_index(drop=True)

    kept = []
    last_end = -1
    for _, r in df.iterrows():
        if int(r["onset_idx"]) <= last_end:
            continue
        kept.append(r)
        last_end = int(r["end_idx"])

    df = pd.DataFrame(kept) if len(kept) else df.iloc[0:0]

    return x_det, df, sigma


# ============================================================
# CLASSI DI AMPIEZZA
# ============================================================

def assign_amplitude_class(amp):
    for label, a0, a1 in AMP_BINS:
        if a0 <= amp < a1:
            return label
    return None


# ============================================================
# WAVEFORM MEDIE
# ============================================================

def build_average_waveform(x_det, subset, fs, stronger_smooth=False):
    pre = int(AVG_PRE_MS / 1000 * fs)
    post = int(AVG_POST_MS / 1000 * fs)
    up = UPSAMPLE_FACTOR

    waves = []

    for _, r in subset.iterrows():
        pk = int(r["peak_idx"])
        if pk - pre < 0 or pk + post >= len(x_det):
            continue

        seg = x_det[pk - pre:pk + post].copy()
        seg = seg - np.median(seg[:pre])

        t0 = np.arange(len(seg)) / fs
        f = interp1d(t0, seg, kind="cubic")
        t_up = np.linspace(t0[0], t0[-1], len(seg) * up)
        seg_up = f(t_up)

        # align to interpolated minimum
        pk_up = np.argmin(seg_up)
        center = pre * up
        shift = center - pk_up

        if shift > 0:
            seg_up = np.pad(seg_up, (shift, 0), mode="edge")[:len(seg_up)]
        elif shift < 0:
            seg_up = np.pad(seg_up, (0, -shift), mode="edge")[-shift:len(seg_up) - shift]

        waves.append(seg_up)

    if not waves:
        return None, None

    W = np.vstack(waves)
    avg = np.mean(W, axis=0)
    t_ms = (np.arange(len(avg)) - pre * up) / (fs * up) * 1000.0

    # repair notch at center
    idx = np.where((t_ms >= -1.2) & (t_ms <= 1.2))[0]
    if len(idx) > 8:
        left = idx[0] - 5
        right = idx[-1] + 5
        avg[idx] = np.interp(
            t_ms[idx],
            [t_ms[left], t_ms[right]],
            [avg[left], avg[right]]
        )

    # visual smoothing only
    if stronger_smooth:
        avg = savgol_filter(avg, 41, 3, mode="interp")
    else:
        avg = savgol_filter(avg, 25, 3, mode="interp")

    return t_ms, avg


# ============================================================
# FINESTRA RAPPRESENTATIVA
# ============================================================

def choose_representative_window(events, x, fs, window_s=30):
    if len(events) == 0:
        return 0

    duration_s = len(x) / fs
    global_freq = len(events) / duration_s
    global_mean_amp = events["amp_pA"].mean()

    win = int(window_s * fs)
    candidates = []

    for start in range(0, len(x) - win + 1, win):
        end = start + win
        sub = events[(events["peak_idx"] >= start) & (events["peak_idx"] < end)]
        if len(sub) == 0:
            continue

        freq = len(sub) / window_s
        mean_amp = sub["amp_pA"].mean()
        score = abs(freq - global_freq) + abs(mean_amp - global_mean_amp)
        candidates.append((start, score))

    if not candidates:
        return 0

    candidates.sort(key=lambda z: z[1])
    return candidates[0][0]


# ============================================================
# FIGURA FINALE
# ============================================================

def make_final_figure(x, x_det, events, fs, outbase):
    x_disp = lowpass(x, fs, DISPLAY_LOWPASS_HZ)

    rep_start = choose_representative_window(events, x, fs, REPRESENTATIVE_WINDOW_S)
    win = int(REPRESENTATIVE_WINDOW_S * fs)

    events = events.copy()
    events["amp_class"] = events["amp_pA"].apply(assign_amplitude_class)
    events = events[events["amp_class"].notna()].copy()

    avg_data = {}
    mins, maxs = [], []

    for cls in ["1", "2", "3"]:
        subset = events[events["amp_class"] == cls]
        stronger = (cls == "1")
        t_ms, avg = build_average_waveform(x_det, subset, fs, stronger_smooth=stronger)
        avg_data[cls] = (t_ms, avg)

        if avg is not None:
            mins.append(np.min(avg))
            maxs.append(np.max(avg))

    common_ymin = min(mins) - 0.5 if mins else -10
    common_ymax = max(maxs) + 0.5 if maxs else 5

    fig = plt.figure(figsize=(12, 6.4), facecolor="white")
    gs = fig.add_gridspec(
        4, 2,
        width_ratios=[2.8, 1.0],
        height_ratios=[1, 1, 1, 0.28],
        hspace=0.18,
        wspace=0.35
    )

    # Left trace
    axL = fig.add_subplot(gs[:3, 0])
    i0 = rep_start
    i1 = rep_start + win
    t = np.arange(i0, i1) / fs
    seg = x_disp[i0:i1] - np.median(x_disp[i0:i1])
    axL.plot(t, seg, color="black", linewidth=0.8)
    axL.axis("off")

    yminL = np.percentile(seg, 0.5) - 1.2
    ymaxL = np.percentile(seg, 99.5) + 1.2
    axL.set_ylim(yminL, ymaxL)

    # Left scale bar in separate row
    axLsb = fig.add_subplot(gs[3, 0])
    axLsb.axis("off")
    axLsb.set_xlim(0, 30)
    axLsb.set_ylim(0, 8)
    axLsb.plot([18, 28], [1.5, 1.5], color="black", linewidth=2)  # 10 s
    axLsb.plot([18, 18], [1.5, 6.5], color="black", linewidth=2)  # 5 pA

    # Right panels
    for i, cls in enumerate(["1", "2", "3"]):
        ax = fig.add_subplot(gs[i, 1])
        t_ms, avg = avg_data[cls]
        if avg is not None:
            ax.plot(t_ms, avg, color="black", linewidth=1.5)

        ax.set_xlim(-10, 40)
        ax.set_ylim(common_ymin, common_ymax)
        ax.axis("off")

        # only bottom right scale
        if cls == "3":
            sb = ax.inset_axes([0.58, -0.24, 0.35, 0.22])
            sb.axis("off")
            sb.set_xlim(0, 14)
            sb.set_ylim(0, 7)
            sb.plot([2, 12], [1.5, 1.5], color="black", linewidth=2)  # 10 ms
            sb.plot([2, 2], [1.5, 6.5], color="black", linewidth=2)   # 5 pA

    png_path = outbase + ".png"
    pdf_path = outbase + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


# ============================================================
# TRACCIATI DA 1 s CON EVENTI
# ============================================================

def plot_first_seconds_with_events(x, events, fs, outdir, n_seconds=20):
    os.makedirs(outdir, exist_ok=True)
    x_disp = lowpass(x, fs, 1000)

    for sec in range(min(n_seconds, int(len(x) // fs))):
        i0 = int(sec * fs)
        i1 = int((sec + 1) * fs)
        t = np.arange(i0, i1) / fs
        seg = x_disp[i0:i1] - np.median(x_disp[i0:i1])

        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.plot(t, seg, color="black", linewidth=1.0)

        sub = events[(events["peak_idx"] >= i0) & (events["peak_idx"] < i1)]
        for _, r in sub.iterrows():
            p = int(r["peak_idx"])
            a = int(r["onset_idx"])
            b = int(r["end_idx"])
            aa = max(i0, a)
            bb = min(i1, b)
            ax.plot(t[aa - i0:bb - i0], seg[aa - i0:bb - i0], color="red", linewidth=1.6)
            ax.scatter([p / fs], [seg[p - i0]], color="red", s=18)

        ax.set_ylim(np.percentile(seg, 1) - 2, np.percentile(seg, 99) + 2)
        ax.set_title(f"second {sec}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("pA")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"sec_{sec:03d}.png"), dpi=220)
        plt.close(fig)


# ============================================================
# ANALISI COMPLETA DI UN FILE
# ============================================================

def analyze_file(path, outdir):
    x, fs, name = load_trace(path)
    x_det, events, sigma = detect_events(x, fs)

    if len(events) == 0:
        print(f"Nessun evento trovato in {name}")
        return

    duration_s = len(x) / fs

    summary = pd.DataFrame([{
        "file": name,
        "duration_s": duration_s,
        "n_events": int(len(events)),
        "frequency_Hz": len(events) / duration_s,
        "mean_amp_pA": events["amp_pA"].mean(),
        "median_amp_pA": events["amp_pA"].median(),
        "mean_charge_pC": events["charge_pC"].mean(),
        "median_charge_pC": events["charge_pC"].median(),
        "mean_rise_ms": events["rise_ms"].mean(),
        "mean_decay_ms": events["decay_ms"].mean(),
        "mean_duration_ms": events["duration_ms"].mean(),
        "noise_sigma_pA": sigma,
    }])

    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    events.to_csv(os.path.join(outdir, "events.csv"), index=False)

    fig_base = os.path.join(outdir, "final_figure")
    make_final_figure(x, x_det, events, fs, fig_base)

    plot_first_seconds_with_events(
        x, events, fs,
        os.path.join(outdir, "seconds"),
        n_seconds=FIRST_SECONDS_TO_PLOT
    )

    print(summary.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for f in FILES:
        if not os.path.exists(f):
            print(f"File non trovato: {f}")
            continue

        name = os.path.splitext(os.path.basename(f))[0]
        outdir = os.path.join(OUTPUT_DIR, name)
        analyze_file(f, outdir)


if __name__ == "__main__":
    main()