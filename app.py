import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit


st.set_page_config(
    page_title="Raman Peak Analyzer",
    layout="wide"
)

st.title("Raman Peak Analyzer")


# ============================================================
# LORENTZIAN
# ============================================================

def lorentzian(x, x0, gamma, A):

    return A * (
        gamma**2 /
        ((x - x0)**2 + gamma**2)
    )


# ============================================================
# RAYLEIGH CORRECTION
# Suppresses the intense elastic scattering peak near 0 cm⁻¹.
# Any signal within `rayleigh_cutoff` cm⁻¹ of zero is zeroed out.
# ============================================================
RAYLEIGH_CUTOFF = 50   # cm⁻¹  — adjust if your notch filter edge differs

def apply_rayleigh_correction(x, y, cutoff=RAYLEIGH_CUTOFF):
    """Zero-out the Rayleigh region (|x| < cutoff)."""
    y_corrected = y.copy().astype(float)
    y_corrected[np.abs(x) < cutoff] = 0.0
    return y_corrected    
# ============================================================
# RAMAN SHIFT DISPLAY RANGE
# ============================================================

X_MIN, X_MAX = 0, 550   # cm⁻¹


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_spectrum(uploaded_file):

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|,|\t",
        engine="python",
        header=None
    )

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # --------------------------------------------------------
    # SHIFT X-AXIS SO RAYLEIGH PEAK (intensity maximum) = 0
    # This is the core calibration from the reference script:
    # find the index of maximum intensity and subtract that
    # x-value from the entire axis, so the elastic scatter
    # peak sits exactly at 0 cm⁻¹.
    # --------------------------------------------------------

    max_idx = np.argmax(y)
    x_use   = x - x[max_idx]       # Rayleigh peak → 0

    # Keep the raw y as-is for the raw panel (before any processing)
    y_raw_original = y.copy()

    # --------------------------------------------------------
    # RETAIN ONLY POSITIVE RAMAN SHIFTS  (Stokes side)
    # --------------------------------------------------------

    pos_mask  = x_use >= 0
    x_use     = x_use[pos_mask]
    y         = y[pos_mask]
    y_raw_pos = y_raw_original[pos_mask]   # raw signal aligned to same mask

    # --------------------------------------------------------
    # RAYLEIGH CORRECTION — zero out residual elastic scatter
    # within RAYLEIGH_CUTOFF cm⁻¹ of the shifted zero
    # --------------------------------------------------------

    y = apply_rayleigh_correction(x_use, y, cutoff=RAYLEIGH_CUTOFF)

    # --------------------------------------------------------
    # SMOOTHING
    # --------------------------------------------------------

    y_smooth = savgol_filter(y, 21, 3)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = savgol_filter(y_smooth, 151, 3)
    signal   = y_smooth - baseline
    signal   = np.clip(signal, 0, None)

    # --------------------------------------------------------
    # NOISE  (estimated from a quiet region beyond Rayleigh)
    # --------------------------------------------------------

    noise_region = signal[(x_use > 700) & (x_use < 1200)]

    if len(noise_region) == 0:
        noise_std = np.std(signal)
    else:
        noise_std = np.std(noise_region)

    dynamic_prominence = 4 * noise_std
    dynamic_height     = 3 * noise_std

    # --------------------------------------------------------
    # CANDIDATE PEAKS
    # --------------------------------------------------------

    candidate_peaks, _ = find_peaks(
        signal,
        prominence=dynamic_prominence,
        height=dynamic_height,
        distance=8,
        width=2
    )

    # Remove low-shift region (beyond Rayleigh cutoff) and
    # restrict to display range
    candidate_peaks = np.array([
        p for p in candidate_peaks
        if RAYLEIGH_CUTOFF < x_use[p] <= X_MAX
    ])

    # --------------------------------------------------------
    # REMOVE FALSE PEAKS  (SNR + width filter)
    # --------------------------------------------------------

    filtered_peaks = []

    for p in candidate_peaks:

        peak_height = signal[p]
        half_height = peak_height / 2

        left = p
        while left > 0 and signal[left] > half_height:
            left -= 1

        right = p
        while right < len(signal) - 1 and signal[right] > half_height:
            right += 1

        width = right - left

        local_region = signal[max(0, p - 20):min(len(signal), p + 20)]
        local_noise  = np.std(local_region)
        snr          = peak_height / (local_noise + 1e-9)

        if snr < 2.5:
            continue
        if width < 3:
            continue

        filtered_peaks.append(p)

    candidate_peaks = np.array(filtered_peaks)

    # --------------------------------------------------------
    # LORENTZIAN FIT — returns position, amplitude, FWHM
    # --------------------------------------------------------

    final_peaks = []

    for p in candidate_peaks:

        try:

            left  = max(0, p - 15)
            right = min(len(signal) - 1, p + 15)

            x_fit = x_use[left:right]
            y_fit = signal[left:right]

            p0 = [x_use[p], 5, signal[p]]

            popt, _ = curve_fit(lorentzian, x_fit, y_fit, p0=p0)

            x0, gamma, A = popt
            fwhm = 2 * abs(gamma)

            # Only keep fits whose centre falls inside display range
            if RAYLEIGH_CUTOFF < x0 <= X_MAX:
                final_peaks.append((x0, A, fwhm))

        except Exception:
            pass

    return {
        "sample": uploaded_file.name,
        "x":      x_use,
        "y_raw":  y_raw_pos,   # shifted + masked raw signal for raw panel
        "signal": signal,
        "peaks":  final_peaks
    }


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Raman Files",
    type=["txt"],
    accept_multiple_files=True
)


if uploaded_files:

    results = []

    with st.spinner("Analyzing Files..."):
        for file in uploaded_files:
            result = analyze_spectrum(file)
            results.append(result)

    st.success(f"{len(results)} files processed.")

    # ========================================================
    # PEAK TABLE  (position | intensity | FWHM)
    # ========================================================

    peak_rows = []

    for result in results:
        sample = result["sample"]
        for i, peak in enumerate(result["peaks"], start=1):
            peak_rows.append({
                "Sample":             sample,
                "Peak Number":        i,
                "Raman Shift (cm⁻¹)": round(peak[0], 2),
                "Intensity":          round(peak[1], 2),
                "FWHM (cm⁻¹)":        round(peak[2], 2)
            })

    peak_df = pd.DataFrame(peak_rows)

    st.subheader("Detected Peaks")
    st.dataframe(peak_df, use_container_width=True)

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv = peak_df.to_csv(index=False)

    st.download_button(
        "Download Peak Table",
        csv,
        file_name="Peak_Table.csv",
        mime="text/csv"
    )

    # ========================================================
    # OVERLAY GRAPH  (processed signals, zoomed 0–550 cm⁻¹)
    # ========================================================

    st.subheader("Overlay Raman Spectra (Processed)")

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        ax.plot(result["x"], result["signal"], label=result["sample"])

    ax.set_xlim(X_MIN, X_MAX)
    ax.legend()
    ax.grid()
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity")

    st.pyplot(fig)
    plt.close(fig)

    # ========================================================
    # RAW + PROCESSED panels — one figure per sample
    # ========================================================

    st.subheader("Raw Data Spectra (per sample)")

    for result in results:

        st.markdown(f"**{result['sample']}**")

        fig_raw, axes = plt.subplots(
            2, 1,
            figsize=(12, 8),
            sharex=True
        )

        # — top panel: baseline-corrected / processed signal ——————————
        axes[0].plot(
            result["x"],
            result["signal"],
            color="steelblue",
            linewidth=1.2
        )

        # Mark detected peaks
        for peak in result["peaks"]:
            axes[0].axvline(
                peak[0],
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=0.8
            )

        axes[0].set_xlim(X_MIN, X_MAX)
        axes[0].set_title("Processed Signal (baseline-corrected, Rayleigh-corrected)")
        axes[0].set_ylabel("Intensity")
        axes[0].set_xlabel("Raman Shift (cm⁻¹)")   # ← x-axis label on processed panel
        axes[0].grid(alpha=0.4)

        # — bottom panel: raw (Rayleigh-corrected) signal —————————————
        raw_x = result["x"] if len(result["x"]) == len(result["y_raw"]) else np.arange(len(result["y_raw"]))

        axes[1].plot(
            raw_x,
            result["y_raw"],
            color="darkorange",
            linewidth=1.0,
            alpha=0.85
        )

        axes[1].set_xlim(X_MIN, X_MAX)
        axes[1].set_title("Raw Signal (Rayleigh-corrected, unprocessed)")
        axes[1].set_ylabel("Intensity")
        axes[1].set_xlabel("Raman Shift (cm⁻¹)")
        axes[1].grid(alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig_raw)
        plt.close(fig_raw)

    # ========================================================
    # PEAK COMPARISON GRAPH — staggered labels, zoomed range
    # ========================================================

    st.subheader("Peak Comparison Graph")

    sample_names = [r["sample"] for r in results]
    y_positions  = list(range(len(sample_names)))

    fig2, ax2 = plt.subplots(
        figsize=(14, max(6, len(results) * 1.4))
    )

    for y_idx, result in enumerate(results):

        peaks = [p[0] for p in result["peaks"]]

        ax2.scatter(peaks, [y_idx] * len(peaks), s=80, zorder=3)

        # Stagger close labels vertically
        prev_x      = -np.inf
        row         = 0
        row_offsets = [0.18, 0.36, -0.18]

        for peak in sorted(peaks):

            if peak - prev_x < 60:
                row = (row + 1) % len(row_offsets)
            else:
                row = 0

            ax2.text(
                peak,
                y_idx + row_offsets[row],
                f"{peak:.0f}",
                fontsize=8,
                ha="center",
                va="bottom"
            )

            prev_x = peak

    ax2.set_xlim(X_MIN, X_MAX)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(sample_names)
    ax2.grid(alpha=0.35)
    ax2.set_xlabel("Raman Shift (cm⁻¹)")
    ax2.set_ylabel("Sample")
    ax2.set_ylim(-0.7, len(results) - 0.3)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
