# BCG-LuminaX

BCG-LuminaX develops sleep staging and respiratory event detection around the local SleepNetZero code under `ahi/`. The original model is described in the [SleepNetZero paper](https://arxiv.org/html/2410.22646v1) and [upstream repository](https://github.com/lszshu/IMWUT2024-SleepNetZero).

Source code, configurations, and compact analysis are tracked in Git. The raw recordings, intermediate signals, model weights, and run logs belong under `/scratch/s5415436/BCG/`. The initial test data consists of two [PhysioNet 2018](https://physionet.org/content/challenge-2018/1.0.0/) polysomnograms.

## Two-record PSG smoke run

Run the sample migration from the project root in a session with writable scratch. It verifies all eight files before deleting their source copies and can be rerun after a completed or interrupted transfer.

```bash
python ahi/move_psg_samples.py
```

The smoke runner defaults to CPU and reads `/scratch/s5415436/BCG/test_data/`. It writes a summary and window logits to a timestamped directory under `/scratch/s5415436/BCG/runs/`. Use a new directory when specifying `--out-dir`.

On Griffith HPC, the existing `graphiti_env` provides PyTorch, Transformers, NumPy, SciPy, and SymPy. Its package set can stay as installed. Use Anaconda base for HDF5 labels if `h5py` is absent from the model environment.

```bash
conda run --no-capture-output -n graphiti_env python ahi/smoke_psg.py \
  --labels-python /sw/anaconda3/2024.06/bin/python
```

Use `--data-dir` to select another directory with the same record layout. On Griffith HPC, request one GPU from the `gpuq2` PBS queue and pass `--device cuda` to the runner inside the job. PBS sets `CUDA_VISIBLE_DEVICES` for the allocated GPU.

In a standalone environment, use Python 3.10+, PyTorch, Transformers, NumPy, SciPy, SymPy, and h5py. The runner uses the `wfdb` package when installed and includes a reader for these two records' local WFDB file layout. The existing Lightning checkpoint-loading path continues to use Lightning.

See the [smoke run results](docs/psg-smoke.md) for validation details and the AHI counting convention.
