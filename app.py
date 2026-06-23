import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Raman Peak Analyzer · 50–500 cm⁻¹",
    layout="wide",
    page_icon="🔬"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #7eb8f7; letter-spacing: 0.04em; }
    h2, h3 { color: #a8d4ff; }
    .stDataFrame { border: 1px solid #2a3a4a; border-radius: 6px; }
    .metric-box {
        background: #151c26;
        border: 1px solid #2a3a4a;
        border-radius: 8px;
        padding: 12px 18px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Raman Peak Analyzer  ·  50–500 cm⁻¹")
st.caption("Adaptive baseline · Flexible peak detection · Per-sample tuning")

# ─────────────────────────────────────────────
# LORENTZIAN
# ─────────────────────────────────────────────
def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


# ─────────────────────────────────────────────
# ADAPTIVE BASELINE  ← key innovation
#
# Strategy: iterative percentile-based baseline.
# In each pass we estimate the baseline as the
# running percentile of the signal; points that
# are much higher than the local background are
# iteratively "pulled down" toward the baseline.
# The window and percentile self-tune to the
# data's peak density.
# ─────────────────────────────────────────────
def adaptive_baseline(y, x, window_pct=0.12, n_iter=10, percentile=20):
    """
    Iterative percentile baseline.
    window_pct : window as fraction of total data length
    n_iter     : refinement iterations
    percentile : which percentile defines 'background'
    """
    n = len(y)
    win = max(int(n * window_pct) | 1, 7)   # must be odd-ish; ensure ≥ 7
    if win % 2 == 0:
        win += 1

    baseline = y.copy().astype(float)

    for _ in range(n_iter):
        smoothed = np.array([
            np.percentile(baseline[max(0, i - win // 2): i + win // 2 + 1], percentile)
            for i in range(n)
        ])
        # Only let baseline go up where signal is near background
        baseline = np.where(y - smoothed < 0.5 * np.std(y - smoothed), smoothed, baseline)

    # Final smooth pass
    baseline = savgol_filter(baseline, min(win, n - 2 if n > 2 else 3), 2)
    return baseline


# ─────────────────────────────────────────────
# ADAPTIVE PEAK DETECTION
#
# Dynamically estimates:
#   • noise floor from quiet regions
#   • prominence threshold from signal distribution
#   • distance from estimated peak density
# ─────────────────────────────────────────────
def detect_peaks_adaptive(signal, x, sensitivity=1.0):
    """
    sensitivity > 1  → detect more (smaller) peaks
    sensitivity < 1  → detect fewer (stronger) peaks
    """
    if len(signal) == 0:
        return []

    # --- noise from lowest-variance quartile of the signal ---
    q25 = np.percentile(signal, 25)
    q75 = np.percentile(signal, 75)
    iqr = q75 - q25
    noise_est = iqr / 1.35          # robust σ estimate

    # --- adaptive thresholds ---
    prom_thresh  = max(noise_est * (3.0 / sensitivity), 1e-6)
    height_thresh = max(noise_est * (2.0 / sensitivity), 1e-6)

    # --- adaptive distance (spread out to avoid packing) ---
    pts_per_unit = len(signal) / (x[-1] - x[0] + 1e-9)
    min_dist_cm = 8 / sensitivity       # in cm⁻¹
    min_dist_pts = max(int(pts_per_unit * min_dist_cm), 3)

    peaks, props = find_peaks(
        signal,
        prominence=prom_thresh,
        height=height_thresh,
        distance=min_dist_pts,
        width=2
    )

    # --- SNR filter ---
    validated = []
    for p in peaks:
        ph = signal[p]
        region = signal[max(0, p - 20): min(len(signal), p + 20)]
        local_noise = np.std(region) + 1e-9
        snr = ph / local_noise
        if snr >= (2.0 / sensitivity):
            validated.append(p)

    return np.array(validated)


# ─────────────────────────────────────────────
# LORENTZIAN FIT
# ─────────────────────────────────────────────
def fit_lorentzian(signal, x_use, p):
    half_win = 20
    left  = max(0, p - half_win)
    right = min(len(signal) - 1, p + half_win)
    x_fit = x_use[left:right]
    y_fit = signal[left:right]
    if len(x_fit) < 5:
        return None
    try:
        p0 = [x_use[p], 5.0, signal[p]]
        popt, _ = curve_fit(lorentzian, x_fit, y_fit, p0=p0, maxfev=3000)
        x0, gamma, A = popt
        if A <= 0 or abs(gamma) > 200:
            return None
        fwhm = 2 * abs(gamma)
        return (x0, A, fwhm)
    except Exception:
        return None


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────
def analyze_spectrum(uploaded_file, sensitivity, baseline_window_pct, baseline_percentile, smooth_window):
    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|,|\t",
        engine="python",
        header=None
    )
    x_raw = df.iloc[:, 0].values.astype(float)
    y_raw = df.iloc[:, 1].values.astype(float)

    # ── restrict to 50–500 cm⁻¹ ──
    mask = (x_raw >= 50) & (x_raw <= 500)
    x = x_raw[mask]
    y = y_raw[mask]

    if len(x) < 20:
        return None, "Too few data points in 50–500 cm⁻¹ range."

    # ── smoothing (window must be odd and < len) ──
    sw = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    sw = min(sw, len(y) - 2 if len(y) > 2 else 3)
    y_smooth = savgol_filter(y, sw, 3)

    # ── adaptive baseline ──
    baseline = adaptive_baseline(
        y_smooth, x,
        window_pct=baseline_window_pct,
        percentile=baseline_percentile
    )
    signal = y_smooth - baseline
    signal = np.clip(signal, 0, None)

    # ── adaptive peak detection ──
    peak_idxs = detect_peaks_adaptive(signal, x, sensitivity=sensitivity)

    # ── Lorentzian fit for each candidate ──
    final_peaks = []
    for p in peak_idxs:
        result = fit_lorentzian(signal, x, p)
        if result is not None:
            final_peaks.append(result)

    return {
        "sample":   uploaded_file.name,
        "x":        x,
        "y_raw":    y,
        "y_smooth": y_smooth,
        "baseline": baseline,
        "signal":   signal,
        "peaks":    final_peaks,
    }, None


# ─────────────────────────────────────────────
# SIDEBAR — controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection Controls")
    st.markdown("---")

    sensitivity = st.slider(
        "Peak Sensitivity",
        min_value=0.3, max_value=3.0, value=1.2, step=0.1,
        help="Higher → detect more / smaller peaks. Lower → only strong peaks."
    )

    st.markdown("**Baseline**")
    baseline_window_pct = st.slider(
        "Baseline Window (% of data)",
        min_value=0.04, max_value=0.40, value=0.12, step=0.01,
        format="%.2f",
        help="Fraction of data length used for background estimation window."
    )
    baseline_percentile = st.slider(
        "Baseline Percentile",
        min_value=5, max_value=40, value=15, step=1,
        help="Lower percentile → more aggressive background removal (useful when peaks are dense)."
    )

    st.markdown("**Smoothing**")
    smooth_window = st.slider(
        "Savitzky-Golay Window (pts)",
        min_value=5, max_value=51, value=11, step=2,
        help="Smoothing window. Larger = smoother but may broaden peaks."
    )

    st.markdown("---")
    st.info("💡 If peaks are missed, increase **Sensitivity**.\n\nIf false peaks appear, decrease it or increase **Baseline Percentile**.")

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload Raman spectra (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.markdown("""
    <div style='text-align:center; padding: 60px 0; color: #4a6080;'>
        <h3>↑ Upload one or more Raman spectrum files to begin</h3>
        <p>Expected format: two columns — Raman shift (cm⁻¹) and intensity</p>
        <p>Analysis window fixed at <strong>50 – 500 cm⁻¹</strong></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────
results = []
errors  = []

with st.spinner("Analysing spectra…"):
    for f in uploaded_files:
        res, err = analyze_spectrum(
            f,
            sensitivity=sensitivity,
            baseline_window_pct=baseline_window_pct,
            baseline_percentile=baseline_percentile,
            smooth_window=smooth_window
        )
        if err:
            errors.append((f.name, err))
        else:
            results.append(res)

if errors:
    for name, msg in errors:
        st.error(f"**{name}**: {msg}")

if not results:
    st.stop()

# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────
total_peaks = sum(len(r["peaks"]) for r in results)
cols = st.columns(3)
cols[0].metric("Files processed", len(results))
cols[1].metric("Total peaks detected", total_peaks)
cols[2].metric("Avg peaks / sample", f"{total_peaks / len(results):.1f}")

st.markdown("---")

# ─────────────────────────────────────────────
# PEAK TABLE
# ─────────────────────────────────────────────
peak_rows = []
for result in results:
    for i, peak in enumerate(result["peaks"], start=1):
        peak_rows.append({
            "Sample":             result["sample"],
            "Peak #":             i,
            "Raman Shift (cm⁻¹)": round(peak[0], 1),
            "Amplitude":          round(peak[1], 1),
            "FWHM (cm⁻¹)":        round(peak[2], 1),
        })

peak_df = pd.DataFrame(peak_rows)

st.subheader("📋 Detected Peaks")
st.dataframe(peak_df, use_container_width=True, height=280)

csv = peak_df.to_csv(index=False)
st.download_button("⬇ Download Peak Table (CSV)", csv, "Peak_Table.csv", "text/csv")

st.markdown("---")

# ─────────────────────────────────────────────
# PER-SAMPLE PLOTS (baseline + signal + peaks)
# ─────────────────────────────────────────────
DARK_BG   = "#0e1117"
PANEL_BG  = "#131a23"
GRID_COL  = "#1e2d3d"
BLUE      = "#5b9bd5"
ORANGE    = "#e07b39"
GREEN     = "#4caf7d"
RED_LINE  = "#e05c5c"
GREY      = "#8899aa"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   "#c8d8e8",
    "xtick.color":       GREY,
    "ytick.color":       GREY,
    "text.color":        "#c8d8e8",
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#131a23",
    "legend.edgecolor":  GRID_COL,
    "font.family":       "DejaVu Sans",
})

st.subheader("📈 Per-Sample Spectra")

for result in results:
    st.markdown(f"### {result['sample']}")

    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 1, hspace=0.42)

    # ── TOP: raw + baseline ──
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(result["x"], result["y_raw"],    color=GREY,   lw=0.9, alpha=0.6, label="Raw")
    ax0.plot(result["x"], result["y_smooth"], color=BLUE,   lw=1.2, label="Smoothed")
    ax0.plot(result["x"], result["baseline"], color=ORANGE, lw=1.4, ls="--", label="Adaptive baseline")
    ax0.set_title("Raw Signal + Adaptive Baseline", pad=8, fontsize=11)
    ax0.set_ylabel("Intensity")
    ax0.set_xlim(50, 500)
    ax0.legend(fontsize=8)
    ax0.grid(True)

    # ── BOTTOM: baseline-corrected + peaks ──
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(result["x"], result["signal"], color=GREEN, lw=1.3, label="Corrected signal")

    for peak in result["peaks"]:
        px, pa, pf = peak
        ax1.axvline(px, color=RED_LINE, lw=0.9, ls="--", alpha=0.7)
        # draw fitted Lorentzian
        x_curve = np.linspace(max(50, px - pf * 3), min(500, px + pf * 3), 200)
        y_curve = lorentzian(x_curve, px, pf / 2, pa)
        ax1.fill_between(x_curve, y_curve, alpha=0.18, color=RED_LINE)
        ax1.text(
            px, pa * 1.04,
            f"{px:.0f}",
            ha="center", va="bottom", fontsize=7.5, color=RED_LINE
        )

    ax1.set_title("Baseline-Corrected Signal + Detected Peaks", pad=8, fontsize=11)
    ax1.set_xlabel("Raman Shift (cm⁻¹)")
    ax1.set_ylabel("Intensity")
    ax1.set_xlim(50, 500)
    ax1.legend(fontsize=8)
    ax1.grid(True)

    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ─────────────────────────────────────────────
# OVERLAY — all processed signals
# ─────────────────────────────────────────────
st.subheader("🔀 Overlay — Corrected Spectra")

COLORS = ["#5b9bd5", "#4caf7d", "#e07b39", "#b07bcc", "#e05c5c",
          "#f0c040", "#7eccc4", "#f07bb0"]

fig2, ax2 = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
ax2.set_facecolor(PANEL_BG)

for i, result in enumerate(results):
    c = COLORS[i % len(COLORS)]
    ax2.plot(result["x"], result["signal"],
             label=result["sample"], color=c, lw=1.2)
    for peak in result["peaks"]:
        ax2.axvline(peak[0], color=c, lw=0.6, ls=":", alpha=0.5)

ax2.set_xlabel("Raman Shift (cm⁻¹)")
ax2.set_ylabel("Intensity")
ax2.set_xlim(50, 500)
ax2.legend(fontsize=8)
ax2.grid(True)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ─────────────────────────────────────────────
# PEAK COMPARISON SCATTER
# ─────────────────────────────────────────────
st.subheader("📊 Peak Comparison Across Samples")

sample_names = [r["sample"] for r in results]
fig3, ax3 = plt.subplots(
    figsize=(14, max(5, len(results) * 1.6)),
    facecolor=DARK_BG
)
ax3.set_facecolor(PANEL_BG)

ROW_OFFSETS = [0.0, 0.22, -0.22, 0.38, -0.38]

for y_idx, result in enumerate(results):
    peaks_sorted = sorted(result["peaks"], key=lambda p: p[0])
    prev_x = -999
    row    = 0

    c = COLORS[y_idx % len(COLORS)]

    for peak in peaks_sorted:
        px, pa, pf = peak
        ax3.scatter(px, y_idx, s=max(40, min(200, pa / 5)), color=c,
                    zorder=3, alpha=0.85)

        if px - prev_x < 30:
            row = (row + 1) % len(ROW_OFFSETS)
        else:
            row = 0

        ax3.text(
            px, y_idx + ROW_OFFSETS[row] + 0.14,
            f"{px:.0f}",
            ha="center", va="bottom", fontsize=7.5, color=c
        )
        prev_x = px

ax3.set_yticks(range(len(sample_names)))
ax3.set_yticklabels(sample_names, fontsize=9)
ax3.set_xlim(50, 500)
ax3.set_ylim(-0.8, len(results) - 0.2)
ax3.set_xlabel("Raman Shift (cm⁻¹)")
ax3.grid(True, axis="x")
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Adaptive baseline: iterative percentile method · "
    "Peak detection: data-driven prominence/noise thresholds · "
    "Fits: Lorentzian · Range: 50–500 cm⁻¹"
)
