# Xbox APU HRTF Converter

A Python utility to convert standard SOFA (Spatially Oriented Format for Acoustics) HRTF datasets into the proprietary 31-tap, 8-bit C-header format expected by the original Xbox APU (NV2A).

## Overview
The original Xbox APU features a hardware HRTF (Head-Related Transfer Function) engine for 3D spatial audio. However, the hardware requires impulse responses to be strictly formatted:
- **31-tap FIR Filter:** The hardware can only process 31 samples per ear.
- **8-bit Signed Integers:** Coefficients must be quantized to `-127` to `127`.
- **ITD Extraction:** The Interaural Time Delay must be separated from the filter and passed to the hardware as a separate delay value.

This script parses high-resolution floating-point `.sofa` files, performs the necessary processing to meet the hardware's strict limitations, and outputs a ready-to-compile `hrtf.h` C-style header.

## Requirements
* Python 3.x
* `h5py` (for reading HDF5 `.sofa` containers)

```bash
pip install h5py
```

## Dataset Recommendation
This script is specifically tuned to parse **Minimum Phase** and **Free Field Compensated** 48kHz `.sofa` files. Minimum Phase is *required* because it packs all acoustic energy into the beginning of the impulse response, allowing us to safely truncate at 31 taps.

## Additional Information
* https://www.sonicom.eu/tools-and-resources/hrtf-dataset/
* https://aes.org/publications/elibrary-page/?id=22128
* https://xboxdevwiki.net/APU
