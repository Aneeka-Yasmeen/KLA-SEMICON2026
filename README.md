# KLA Restoration -- Submission Package

Blind restoration of grayscale images degraded by Gaussian noise, speckle noise, and 2x
downsampling (unknown combination/order), for the KLA / SEMICON India Hackathon 2026 problem
statement.

Single `run.py` entry point, `.npy`-only input/output, no internet access or manual configuration.

## Quick start

```bash
git clone https://github.com/Aneeka-Yasmeen/KLA-SEMICON2026
cd KLA-SEMICON2026

python -m venv .venv
.venv\Scripts\activate              # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

pip uninstall torch -y
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python run.py <input-dir> <output-dir>
```

See **Setup** and **GPU detection** below for details on the CUDA install step.

## Structure

```
run.py                  <- entry point
requirements.txt
README.md
models/
    __init__.py
    restoration_net.py    <- architecture (Severity-Gated NAF-Bottleneck, 3,013,957 params)
    final_model.pth         <- trained weights (config embedded in the checkpoint)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

No internet access is needed at run time -- the model weights ship in `models/final_model.pth`.
No API keys, downloads, or manual configuration.

**GPU users read this first.** `pip install -r requirements.txt` installs the CPU-only build of
PyTorch by default -- the CUDA build comes from a separate package index. `run.py` still runs
correctly on CPU (it falls back automatically and prints a warning), but a machine with a GPU
will otherwise get scored on CPU throughput. To get the GPU build:

```bash
pip uninstall torch -y
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

Replace `cu126` with whatever matches the target machine's CUDA version -- see
https://pytorch.org/get-started/locally/ for the right index URL. CUDA 12.6 supports H100
(Hopper) as well as older GPU generations.

Verify it worked:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
This should print `True` before running `run.py` if a GPU is expected.

## GPU detection

`run.py` detects whatever GPU is present at runtime via `torch.cuda.is_available()` and
`torch.cuda.get_device_name(0)` -- no hardcoded device name or index. It prints whichever GPU it
found and uses it:
```
[run] device=cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
```
No flag or code change is needed to select the GPU. The only manual step is the CUDA-build
install above, which applies to any PyTorch project, not just this one.

## Run

```bash
python run.py <input-dir> <output-dir>
```

Example:
```bash
python run.py C:\path\to\degraded_npy_files C:\path\to\restored_output
```

Also accepts `--input_dir`/`--output_dir` flags (`python run.py --input_dir <in> --output_dir
<out>`).

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` if it doesn't exist.
- Writes one restored `.npy` file per input, same filename.
- Output is single-channel grayscale, shape `(H, W)`, values clipped to `[0,1]`, no NaN/Inf.
- Output resolution is 2x the input resolution.
- Uses GPU automatically when available, falls back to CPU otherwise.

## Robustness

A single corrupted or unreadable input file does not stop the run or affect other files:
- An unreadable file is logged and skipped; the rest still get restored.
- A file with NaN/Inf values has them replaced with a safe fallback before restoring.
- A batch that runs out of GPU memory is retried at a smaller size, down to single-image.
- A summary at the end of every run lists what happened to every input file.

## Results

Best checkpoint (epoch 115), measured on a held-out 320-sample validation split (10% of the
3,200-pair training set, seeded, no leakage):

| | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic upsampling (baseline) | 22.96 dB | 0.542 | 0.440 |
| **This model** | **28.83 dB** | **0.787** | **0.229** |

3,013,957 parameters, ~100.9 GMACs (lower-bound estimate) at 256x256. RTX 4060 Laptop GPU
runtime: ~8.2-8.9s end-to-end for 400 test images (~45-49 images/sec), I/O included.

## Experimental journey

1. **Smaller model first** (697K params) -- NAFNet-style gated conv blocks, implicit severity
   embedding, one global-context block, PixelShuffle 2x upsampling. 28.46 dB PSNR.
2. **Round 1 ablation** -- removing the global-context block, removing the edge-preservation
   loss, and tripling the edge-loss weight were all tried and all made results worse.
3. **Capacity scaling** (697K -> 2.83M params) -- modest gain (28.46 -> 28.76 dB) at roughly
   2.3x the inference cost.
4. **Round 2 ablation** -- tested a frequency-domain (FFT) loss term against severity-gated
   residual scaling (each block's residual strength predicted from the severity embedding instead
   of a fixed constant). The frequency loss helped alone but added little once severity-gating
   was active. Severity-gated residuals improved PSNR/SSIM/LPIPS at the same parameter count as
   the capacity-scaled model, with no measured increase in inference latency.
5. **This model** -- capacity-scaled architecture plus severity-gated residuals, retrained at
   full budget: 3,013,957 params, 28.83 dB PSNR / 0.787 SSIM / 0.229 LPIPS.

**Known limitation:** one validation sample shows persistent over-smoothing (fine granular
texture flattened, PSNR looks fine but SSIM is low). It survived every ablation round and
capacity increase, which points to a loss-recipe limitation rather than a capacity one. Still
unresolved.

Architectures considered and not used: Mamba/state-space blocks (custom CUDA-extension
dependency, marginal benefit at this image size), full window attention throughout the network
(SwinIR-style -- real latency cost for marginal gain over a CNN at a fixed 2x factor), and
learned prompt banks (PromptIR-style -- built for many degradation types; the KLA task only has
three known ones).

## Notes

- Dependencies are limited to `torch` and `numpy` to match the submission requirements. Training
  code and the full ablation records live in a separate development repository, not included
  here.
- Model: Severity-Gated NAF-Bottleneck, 3,013,957 parameters, ~100.9 GMACs (lower-bound estimate)
  at 256x256.
