# KLA Restoration -- Submission Package

AI-based restoration of degraded images (Gaussian noise + speckle noise + 2x downsampling, in
unknown combination/order) for the KLA / SEMICON India Hackathon 2026 problem statement.

This package matches the final-submission-check requirements exactly: a single `run.py` entry
point, `.npy`-only input/output, no internet access or manual configuration needed.

## Structure

```
run.py                  <- entry point
requirements.txt
README.md
models/
    __init__.py
    restoration_net.py    <- architecture (Severity-Gated NAF-Bottleneck, 3,013,957 params)
    io_utils.py            <- .npy loading/sanitization helpers used by run.py
    final_model.pth         <- trained weights (self-describing: includes the architecture config)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

No internet access is required at run time -- both dependencies are installed once via pip, and
the model weights are already included in `models/final_model.pth`. No API keys, no additional
downloads, no user interaction, no manual configuration.

## Run

```bash
python run.py <input-dir> <output-dir>
```

Example:
```bash
python run.py C:\path\to\degraded_npy_files C:\path\to\restored_output
```

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` automatically if it does not already exist.
- Writes exactly one restored `.npy` file per input file, under the same filename.
- Every output is a single-channel grayscale array, shape `(H, W)`, values clipped to `[0,1]`,
  guaranteed free of NaN/Inf (checked and enforced explicitly, not assumed).
- Output resolution is exactly 2x the input resolution.
- Runs on an NVIDIA GPU automatically when available (falls back to CPU otherwise) -- no flag
  or configuration needed to select the device.

## Robustness

A single corrupted, unreadable, or otherwise problematic input file does not stop the run or
cause other files to be skipped:
- An unreadable file is logged and skipped; every other file still gets restored.
- A file containing NaN/Inf values has them replaced with a safe fallback before restoring
  (logged separately, since real output is still produced for that file).
- A batch that runs out of GPU memory is automatically retried at a smaller size, down to
  single-image, before that one file is given up on.
- A summary is printed at the end of every run listing exactly what happened to every input
  file -- nothing fails silently.

## Notes

- This package is intentionally minimal (only `torch` and `numpy` as dependencies) to match the
  submission-check requirements exactly. The full development repository -- including training
  code, the research/design record, ablation studies, and additional documentation -- is
  available separately at `[GitHub repository URL -- fill in]`.
- Model: Severity-Gated NAF-Bottleneck, 3,013,957 parameters, ~100.9 GMACs (lower-bound estimate)
  at 256x256. Validation-split result: PSNR 28.83dB, SSIM 0.7868, LPIPS 0.229.
