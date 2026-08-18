"""
KLA restoration -- required submission entry point.

Usage (exactly as specified in the final-submission-check requirements):
    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir>, restores it, and writes one .npy file per input to
<output-dir> under the same filename. No internet access, API keys, additional model downloads,
user interaction, or manual configuration are required -- the model architecture and weights are
loaded entirely from the local models/ folder next to this script.

Guarantees, each satisfied deliberately, not by accident:
    - Reads all .npy files from the input directory.
    - Creates the output directory if it does not already exist.
    - Generates exactly one output .npy per input .npy, same filename.
    - Every output is a single-channel grayscale array, shape (H, W).
    - Every output value is clipped to [0,1] and contains no NaN or Inf (checked explicitly,
      not assumed -- see the final sanitization pass below).
    - Output resolution is exactly 2x the input resolution (this model's fixed, officially
      specified scale factor).
    - Runs on an NVIDIA GPU automatically when available, falls back to CPU otherwise.

One bad input file (corrupted, unreadable, containing NaN/Inf, or too large for available GPU
memory) does not crash the run or silently drop other files -- it is logged and skipped, and a
summary at the end reports exactly what happened to every file.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.restoration_net import build_model_from_config
from models.io_utils import infer_hw_from_shape, load_npy_grayscale, sanitize_finite

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "final_model.pth"


def parse_args():
    # Accepts BOTH the required positional form (python run.py <input-dir> <output-dir>) AND
    # the --input_dir/--output_dir flag form used elsewhere in this project's other scripts --
    # deliberately, not by oversight: the two official instruction sources for this competition
    # disagree on which style is required, and a real test run showed this is an easy mistake to
    # make even when you already know the right answer. Supporting both costs nothing and removes
    # a real failure mode instead of just documenting around it.
    parser = argparse.ArgumentParser(description="KLA restoration inference.")
    parser.add_argument("input_dir_positional", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("output_dir_positional", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing degraded .npy files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write restored .npy files to.")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    input_dir = args.input_dir or args.input_dir_positional
    output_dir = args.output_dir or args.output_dir_positional
    if input_dir is None or output_dir is None:
        parser.error(
            "input and output directories are required, either positionally "
            "(python run.py <input-dir> <output-dir>) or as flags "
            "(python run.py --input_dir <input-dir> --output_dir <output-dir>)."
        )
    args.input_dir = input_dir
    args.output_dir = output_dir
    return args


def resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device):
    checkpoint = torch.load(DEFAULT_WEIGHTS_PATH, map_location=device, weights_only=False)
    model = build_model_from_config({"model": checkpoint["model_config"]})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def group_by_shape(files, failures):
    groups = {}
    for f in files:
        try:
            arr = np.load(f, mmap_mode="r")
            shape = infer_hw_from_shape(tuple(arr.shape))
        except Exception as e:
            failures.append((f, f"could not read shape: {e}"))
            continue
        groups.setdefault(shape, []).append(f)
    return groups


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


def run_model_oom_safe(model, batch_files, batch_tensor, device, failures):
    """Runs the model on a batch; on CUDA out-of-memory, halves the batch and retries down to
    single-image before giving up on that one file -- everything else still gets processed."""
    try:
        with torch.no_grad():
            output = model(batch_tensor)
        return batch_files, output.squeeze(1).to("cpu").numpy()
    except torch.cuda.OutOfMemoryError as e:
        if device == "cuda":
            torch.cuda.empty_cache()
        if len(batch_files) == 1:
            failures.append((batch_files[0], f"out of GPU memory even alone: {e}"))
            return [], np.empty((0,) + tuple(batch_tensor.shape[-2:]), dtype=np.float32)
        mid = len(batch_files) // 2
        f_a, a_a = run_model_oom_safe(model, batch_files[:mid], batch_tensor[:mid], device, failures)
        f_b, a_b = run_model_oom_safe(model, batch_files[mid:], batch_tensor[mid:], device, failures)
        files = f_a + f_b
        arr = np.concatenate([a_a, a_b], axis=0) if files else np.empty((0,) + tuple(batch_tensor.shape[-2:]), dtype=np.float32)
        return files, arr


def finalize_output(arr):
    """The required output guarantee, enforced explicitly rather than assumed: single-channel
    (H, W), clipped to [0,1], zero NaN/Inf. Even though the model's sigmoid output and the input
    sanitization make non-finite values very unlikely in practice, this is a hard requirement per
    the submission check, so it is checked and fixed here unconditionally, not trusted blindly."""
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr, _ = sanitize_finite(arr)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return arr


def main():
    args = parse_args()
    device = resolve_device()
    print(f"[run] device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print(
            "[run] WARNING: no CUDA GPU detected -- running on CPU, which will be significantly "
            "slower. If this machine has an NVIDIA GPU, this usually means the CPU-only build of "
            "PyTorch was installed (the default from `pip install -r requirements.txt`). See "
            "README.md's Setup section for the CUDA-specific install command. This is not an "
            "error and the run will complete correctly either way, but throughput will not "
            "reflect GPU performance."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory {input_dir} does not exist.")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy")
    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}.")
    print(f"[run] found {len(files)} input .npy file(s)")

    model = load_model(device)

    failures = []
    sanitized_notes = []
    groups = group_by_shape(files, failures)
    if failures:
        print(f"[run] warning: {len(failures)} file(s) could not be read and will be skipped:")
        for f, reason in failures:
            print(f"    {f.name}: {reason}")

    total_start = time.perf_counter()
    num_processed = 0

    for shape, group_files in groups.items():
        for batch_files in chunk(group_files, args.batch_size):
            arrays, loaded_files = [], []
            for f in batch_files:
                try:
                    arr = load_npy_grayscale(f)
                    arr, n_bad = sanitize_finite(arr)
                    if n_bad:
                        sanitized_notes.append((f.name, f"{n_bad} non-finite input value(s) replaced"))
                    arrays.append(arr)
                    loaded_files.append(f)
                except Exception as e:
                    failures.append((f, f"failed to load: {e}"))
            if not loaded_files:
                continue

            batch = torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(1).to(device, non_blocking=True)
            saved_files, output_np = run_model_oom_safe(model, loaded_files, batch, device, failures)

            for f, out_arr in zip(saved_files, output_np):
                out_arr = finalize_output(out_arr)
                np.save(output_dir / f.name, out_arr)
            num_processed += len(saved_files)

    total_time_s = time.perf_counter() - total_start
    print(f"[run] processed {num_processed}/{len(files)} image(s) in {total_time_s:.3f}s end-to-end "
          f"(read + preprocess + transfer + compute + postprocess + save)")
    if total_time_s > 0:
        print(f"[run] throughput = {num_processed / total_time_s:.2f} images/sec")

    if sanitized_notes:
        print(f"[run] note: {len(sanitized_notes)} file(s) had non-finite input values, sanitized before restoring:")
        for name, reason in sanitized_notes:
            print(f"    {name}: {reason}")
    if failures:
        print(f"[run] WARNING: {len(failures)} file(s) produced NO output:")
        for f, reason in failures:
            print(f"    {f.name}: {reason}")
        print(f"[run] {num_processed}/{len(files)} input file(s) restored successfully; "
              f"the rest are listed above and were skipped, not silently dropped.")


if __name__ == "__main__":
    main()
