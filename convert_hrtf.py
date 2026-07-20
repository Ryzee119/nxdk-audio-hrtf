#!/usr/bin/env python3
"""
SONICOM HRTF → Xbox APU Converter

Converts a SONICOM FreeFieldCompMinPhase_48kHz SOFA file into a C header
file for the Xbox APU's hardware HRTF engine.

The SONICOM FreeFieldCompMinPhase_48kHz data provides:
  - 48kHz sample rate (native match for Xbox APU — no resampling needed)
  - Pre-computed minimum-phase HRIRs (energy packed in early taps)
  - Pre-computed ITD stored as metadata (no onset detection needed)
  - Free-field compensated (measurement artifacts removed)

Requirements:
  pip install h5py

Usage:
  python3 convert_hrtf.py <sofa_file> [--output nxaudio_hrtf.h]

Download SOFA files from:
  https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/
  Navigate to: XXXXXX/HRTF/HRTF/48kHz/XXXXX_FreeFieldCompMinPhase_48kHz.sofa
"""

import argparse
import math
import os
import sys

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)


# ============================================================================
# Constants
# ============================================================================

# Xbox APU hardware parameters
NUM_FIR_TAPS = 31           # Hardware FIR filter length
MAX_ITD = 42                # Maximum ITD value (hardware limit)

# Output grid (must match nxdk-audio's nxAudioHRTFSetParamsFromAngles)
AZIMUTH_MIN = 0
AZIMUTH_MAX = 180
AZIMUTH_STEP = 5            # 37 azimuth entries (0, 5, 10, ..., 180)

ELEVATION_MIN = -40
ELEVATION_MAX = 90
ELEVATION_STEP = 10         # 14 elevation entries (-40, -30, ..., +90)

NUM_AZIMUTHS = (AZIMUTH_MAX - AZIMUTH_MIN) // AZIMUTH_STEP + 1    # 37
NUM_ELEVATIONS = (ELEVATION_MAX - ELEVATION_MIN) // ELEVATION_STEP + 1  # 14


# ============================================================================
# SOFA file reading
# ============================================================================

def read_sofa(filepath):
    """Read a SOFA file and extract HRIRs, delays, and source positions.

    SOFA (Spatially Oriented Format for Acoustics) files are HDF5 containers
    with standardized dataset names:
      - Data.IR:          [M x R x N]  impulse responses
      - Data.Delay:       [M x R] or [1 x R]  onset delays per receiver
      - SourcePosition:   [M x 3]  azimuth, elevation, distance
      - Data.SamplingRate: scalar

    Where M=measurements, R=receivers (2 for binaural), N=IR samples.
    """
    print(f"Reading SOFA file: {os.path.basename(filepath)}")

    with h5py.File(filepath, 'r') as f:
        ir_data = f['Data.IR'][:]
        delay_data = f['Data.Delay'][:]
        positions = f['SourcePosition'][:]

        sr = f['Data.SamplingRate'][()]
        if hasattr(sr, '__len__'):
            sr = float(sr[0])
        else:
            sr = float(sr)

    num_measurements = ir_data.shape[0]
    num_receivers = ir_data.shape[1]
    ir_length = ir_data.shape[2]

    azimuths = sorted(set(int(round(p)) for p in positions[:, 0]))
    elevations = sorted(set(int(round(p)) for p in positions[:, 1]))

    print(f"  Measurements: {num_measurements}")
    print(f"  Receivers: {num_receivers}")
    print(f"  IR length: {ir_length} samples")
    print(f"  Sample rate: {int(sr)} Hz")
    print(f"  Delay shape: {delay_data.shape}")
    print(f"  Azimuths: {min(azimuths)}° to {max(azimuths)}° ({len(azimuths)} values)")
    print(f"  Elevations: {min(elevations)}° to {max(elevations)}° ({len(elevations)} values)")

    if int(sr) != 48000:
        print(f"  WARNING: Expected 48kHz, got {int(sr)}Hz. Results may be suboptimal.")

    return ir_data, delay_data, positions


# ============================================================================
# Signal processing
# ============================================================================

def find_onset(ir, threshold_ratio=0.05):
    """Find the onset sample of an impulse response.

    Returns the index of the first sample exceeding threshold_ratio * peak.
    Uses a lower threshold than SOFA (0.05 vs 0.1) because SONICOM
    free-field compensated data has a more gradual onset.
    """
    peak = max(abs(s) for s in ir)
    if peak < 1e-15:
        return 0

    threshold = peak * threshold_ratio
    for i, s in enumerate(ir):
        if abs(s) > threshold:
            return max(0, i - 1)  # Back up one sample to not clip the onset
    return 0


def truncate_and_window(ir, num_taps):
    """Truncate IR to num_taps and apply a half-Hanning fade-out window.

    The window tapers the last quarter of the taps to zero to prevent
    ringing artifacts from the abrupt truncation.
    """
    if len(ir) < num_taps:
        result = list(ir) + [0.0] * (num_taps - len(ir))
    else:
        result = list(ir[:num_taps])

    # Half-Hanning fade-out on the last quarter
    fade_start = num_taps * 3 // 4
    fade_len = num_taps - fade_start

    for i in range(fade_len):
        window = 0.5 * (1.0 + math.cos(math.pi * i / fade_len))
        result[fade_start + i] *= window

    return result


def quantize_ir(taps):
    """Quantize an impulse response to signed 8-bit.

    Each ear is quantized independently so both use the full [-127, +127]
    range, preserving the spectral shape of each ear's HRIR.
    """
    peak = max(abs(s) for s in taps)
    if peak < 1e-10:
        return [0] * len(taps)
    scale = 127.0 / peak
    return [max(-127, min(127, int(round(s * scale)))) for s in taps]


def int8_to_uint8_hex(val):
    """Convert signed int8 to uint8 hex string (two's complement)."""
    if val < 0:
        val += 256
    return f"0x{val:02X}"


# ============================================================================
# Grid lookup and table building
# ============================================================================

def find_nearest_measurement(positions, sofa_azimuth, target_elevation):
    """Find the index of the nearest SOFA measurement to target angles.

    Handles azimuth wrapping (0° and 360° are the same direction).
    """
    best_idx = 0
    best_dist = float('inf')

    for i in range(len(positions)):
        azim = positions[i, 0]
        elev = positions[i, 1]

        # Angular distance with wrapping
        d_azim = abs(azim - sofa_azimuth)
        if d_azim > 180:
            d_azim = 360 - d_azim

        d_elev = abs(elev - target_elevation)
        dist = d_azim ** 2 + d_elev ** 2

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def build_filter_table(ir_data, delay_data, positions):
    """Build the complete filter table for all azimuth/elevation combinations.

    SOFA coordinate convention:
      - Azimuth 0° = front, 90° = LEFT, 270° = RIGHT
      - Our convention: azimuth 0-180 where positive = RIGHT

    For our table at azimuth A (source to the right):
      - Look up SOFA measurement at azimuth (360-A)%360

    Returns:
      - filters: list of (coeffs, delay) tuples (alternating left/right)
      - filter_index: dict (azim_deg, elev_deg) -> index into filters list
    """
    filters = []
    filter_index = {}

    total = NUM_AZIMUTHS * NUM_ELEVATIONS
    processed = 0

    for az_idx in range(NUM_AZIMUTHS):
        our_azimuth = AZIMUTH_MIN + az_idx * AZIMUTH_STEP

        # Convert our azimuth (right=positive) to SOFA azimuth (left=positive)
        # Our 90° (right) = SOFA 270°
        sofa_azimuth = (360 - our_azimuth) % 360

        for el_idx in range(NUM_ELEVATIONS):
            target_elevation = ELEVATION_MIN + el_idx * ELEVATION_STEP

            processed += 1
            if processed % 100 == 0:
                print(f"  Processing {processed}/{total}...")

            # Find nearest SOFA measurement
            m_idx = find_nearest_measurement(positions, sofa_azimuth, target_elevation)

            # Extract left and right ear IRs
            # SOFA: receiver 0 = left ear, receiver 1 = right ear
            left_ear_ir = list(ir_data[m_idx, 0, :])
            right_ear_ir = list(ir_data[m_idx, 1, :])

            # The SONICOM FreeFieldCompMinPhase data still has significant
            # propagation delay (~40 samples of near-silence) before the
            # main energy. We MUST strip this or our 31-tap window captures
            # nothing useful.
            #
            # 1. Detect onset of each ear independently
            # 2. Compute ITD from onset difference
            # 3. Align each ear to its own onset (strip leading silence)
            # 4. Then truncate to 31 taps

            left_onset = find_onset(left_ear_ir, threshold_ratio=0.05)
            right_onset = find_onset(right_ear_ir, threshold_ratio=0.05)

            # ITD from onset difference (positive = left delayed = source right)
            itd = left_onset - right_onset
            itd = max(-MAX_ITD, min(MAX_ITD, itd))

            # Align each ear to its own onset
            aligned_left = left_ear_ir[left_onset:]
            aligned_right = right_ear_ir[right_onset:]

            # Truncate to 31 taps with windowing
            left_taps = truncate_and_window(aligned_left, NUM_FIR_TAPS)
            right_taps = truncate_and_window(aligned_right, NUM_FIR_TAPS)

            # Quantize each ear independently
            left_q = quantize_ir(left_taps)
            right_q = quantize_ir(right_taps)

            # Store as a left/right pair
            pair_index = len(filters)

            even_delay = max(0, min(255, abs(itd))) if itd > 0 else 0
            odd_delay = max(0, min(255, abs(itd))) if itd < 0 else 0

            filters.append((left_q, even_delay))    # even = left ear (contralateral)
            filters.append((right_q, odd_delay))    # odd  = right ear (ipsilateral)
            filter_index[(our_azimuth, target_elevation)] = pair_index

    return filters, filter_index


# ============================================================================
# Output generation
# ============================================================================

def write_header_file(filepath, filters, filter_index, sofa_filename):
    """Write a single .h file containing both filter data and index table."""

    with open(filepath, "w") as f:
        f.write("/*\n")
        f.write(" * SONICOM HRTF data (FreeFieldCompMinPhase, 48kHz) for Xbox APU.\n")
        f.write(" *\n")
        f.write(f" * Source: {sofa_filename}\n")
        f.write(" * Dataset: https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/\n")
        f.write(" * Information: https://aes.org/publications/elibrary-page/?id=22128\n")
        f.write(" * License: Creative Commons Attribution 4.0 (CC BY 4.0)\n")
        f.write(" *\n")
        f.write(" * Generated by convert_hrtf.py\n")
        f.write(" */\n\n")

        f.write("#ifndef NXAUDIO_HRTF_H\n")
        f.write("#define NXAUDIO_HRTF_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("/* 31-tap FIR filter + interaural time delay */\n")
        f.write("typedef struct {\n")
        f.write("    uint8_t coeff[31];\n")
        f.write("    uint8_t delay;\n")
        f.write("} nxaudio_hrtf_filter_t;\n\n")

        num_filters = len(filters)
        f.write(f"#define NXAUDIO_HRTF_NUM_FILTERS   {num_filters}\n")
        f.write(f"#define NXAUDIO_HRTF_NUM_AZIMUTHS  {NUM_AZIMUTHS}"
                f"  /* {AZIMUTH_MIN} to {AZIMUTH_MAX}"
                f" in {AZIMUTH_STEP}-degree steps */\n")
        f.write(f"#define NXAUDIO_HRTF_NUM_ELEVATIONS {NUM_ELEVATIONS}"
                f" /* {ELEVATION_MIN} to {ELEVATION_MAX}"
                f" in {ELEVATION_STEP}-degree steps */\n")
        f.write(f"#define NXAUDIO_HRTF_AZIMUTH_STEP  {AZIMUTH_STEP}\n")
        f.write(f"#define NXAUDIO_HRTF_ELEVATION_STEP {ELEVATION_STEP}\n\n")

        f.write("/*\n")
        f.write(" * Filter coefficients stored as consecutive left/right pairs.\n")
        f.write(" * Coefficients are signed 8-bit in two's complement (stored as uint8_t).\n")
        f.write(" * Delay is the ITD in samples at 48kHz for the delayed ear.\n")
        f.write(" */\n")
        f.write("static const nxaudio_hrtf_filter_t nxaudio_hrtf_filters[] = {\n")

        # Column header
        tap_header = "/* tap:"
        for t in range(NUM_FIR_TAPS):
            tap_header += f" {t:4d} "
        tap_header += "  ITD */"
        f.write(tap_header + "\n")

        # Write filter data
        for i, (coeffs, delay) in enumerate(filters):
            coeff_str = ", ".join(int8_to_uint8_hex(c) for c in coeffs)
            f.write(f"    {{ {{ {coeff_str} }}, {delay:2d} }},"
                    f" /* {i:4d} */\n")

        f.write("};\n\n")

        # Write index table
        f.write("/*\n")
        f.write(" * Index table: nxaudio_hrtf_index[azimuth_idx][elevation_idx]\n")
        f.write(" * Maps grid position to index into nxaudio_hrtf_filters[].\n")
        f.write(" * Each entry points to a left/right filter pair (2 consecutive entries).\n")
        f.write(" */\n")
        f.write(f"static const uint16_t nxaudio_hrtf_index"
                f"[{NUM_AZIMUTHS}][{NUM_ELEVATIONS}] = {{\n")

        f.write(f"  /* elevation= ")
        for el_idx in range(NUM_ELEVATIONS):
            elev_deg = ELEVATION_MIN + el_idx * ELEVATION_STEP
            f.write(f"   {elev_deg:3d}")
        f.write("  */\n")

        for az_idx in range(NUM_AZIMUTHS):
            azim_deg = AZIMUTH_MIN + az_idx * AZIMUTH_STEP
            f.write(f"  /* az={azim_deg:3d}° */ {{ ")

            entries = []
            for el_idx in range(NUM_ELEVATIONS):
                elev_deg = ELEVATION_MIN + el_idx * ELEVATION_STEP
                idx = filter_index.get((azim_deg, elev_deg), 0)
                entries.append(f"{idx:4d}")

            f.write(", ".join(entries))
            f.write(" },\n")

        f.write("};\n\n")
        f.write("#endif /* NXAUDIO_HRTF_H */\n")

    print(f"  Wrote {filepath}")
    print(f"    {num_filters} filter entries, {NUM_AZIMUTHS}x{NUM_ELEVATIONS} index table")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert SONICOM HRTF (SOFA) to Xbox APU C header",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Download SOFA files from the SONICOM transfer portal:
  https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/

Navigate to: P00XX/HRTF/HRTF/48kHz/P00XX_FreeFieldCompMinPhase_48kHz.sofa

Example:
  python3 convert_hrtf.py P0001_FreeFieldCompMinPhase_48kHz.sofa -o nxaudio_hrtf.h
""")
    parser.add_argument("sofa_file", help="Path to SONICOM .sofa file")
    parser.add_argument("--output", "-o", default="nxaudio_hrtf.h",
                        help="Output header file path (default: nxaudio_hrtf.h)")
    args = parser.parse_args()

    if not os.path.exists(args.sofa_file):
        print(f"ERROR: File not found: {args.sofa_file}")
        print("\nDownload a SOFA file from:")
        print("  https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/")
        print("  Navigate to: P00XX/HRTF/HRTF/48kHz/P00XX_FreeFieldCompMinPhase_48kHz.sofa")
        sys.exit(1)

    print("=" * 70)
    print("SONICOM HRTF -> Xbox APU Converter")
    print("=" * 70)

    # Read SOFA file
    ir_data, delay_data, positions = read_sofa(args.sofa_file)

    # Build filter table
    print("\nProcessing impulse responses...")
    print(f"  Truncating to {NUM_FIR_TAPS} taps (with Hanning window)")
    print(f"  Quantizing: float -> 8-bit")
    print(f"  Output grid: {NUM_AZIMUTHS} azimuths x {NUM_ELEVATIONS} elevations")

    filters, filter_index = build_filter_table(ir_data, delay_data, positions)

    # Write output
    print("\nWriting output...")
    sofa_filename = os.path.basename(args.sofa_file)
    write_header_file(args.output, filters, filter_index, sofa_filename)

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"  {args.output}")
    print(f"\nTo use: #include this file in your C code.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
