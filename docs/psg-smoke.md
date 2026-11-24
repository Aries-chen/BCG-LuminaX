# Two-record SleepNetZero PSG smoke run

Both records completed the full ingest and alignment checks. For each record, the staging and event heads ran a forward pass, produced a finite loss and finite gradients, and updated an output-layer parameter in one FP32 Adam step on CPU.

| Record | Complete / scored 30 s epochs | Whole-record respiratory events | Event starts in scored sleep | Sleep hours | Annotation AHI (events/h) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tr03-0005` | 857 / 845 | 43 | 36 | 3.9833 | 9.04 |
| `tr03-0029` | 795 / 777 | 153 | 147 | 3.8333 | 38.35 |

| Record | Staging window start (s) | Staging loss | Event window start (s) | Positive event seconds | Event loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tr03-0005` | 360 | 1.60297 | 360 | 17 | 0.75732 |
| `tr03-0029` | 540 | 1.73639 | 929 | 1 | 0.79671 |

The windows are 300 seconds long. Staging produced `[1, 10, 5]` logits from three `[1, 1200]` inputs. The event model produced `[1, 300]` logits from `[1, 15000]` ABD and `[1, 75000]` AIRFLOW inputs. These PSG signals fill the existing `heartbreath` and `snore` input slots respectively. Staging used ECG-derived RR intervals, bandpassed ABD respiration, and a movement indicator based on O1-M2 peak-to-peak amplitude. The ECG and movement extractors are smoke-run implementations.

The reference AHI counts annotated apnea and hypopnea starts during scored sleep divided by scored sleep hours. Sleep time is derived from the sample-level stage masks. For sequence training, a second is marked positive when any part of an event overlaps it. The first valid event window in `tr03-0029` has a single positive second, and its staging window contains only wake epochs. The losses describe an optimization step with random weights and seed 42. They do not measure model performance.

## Reproduction

The verified model environment was the existing `graphiti_env` with Python 3.10.19, PyTorch 2.6.0+cu124, Transformers 4.57.3, NumPy 2.1.3, and SciPy 1.15.3. This run used the local WFDB layout reader. The separate Anaconda base Python 3.12.4 read HDF5 with h5py 3.11.0. No packages in `graphiti_env` were changed.

```bash
run_root=$(mktemp -d /tmp/bcg-psg-smoke.XXXXXX)
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  conda run --no-capture-output -n graphiti_env python -B ahi/smoke_psg.py \
  --data-dir fm_samples --out-dir "$run_root/run" --device cpu \
  --labels-python /sw/anaconda3/2024.06/bin/python
```

The validation session exposed CPU only and mounted `/scratch/s5415436` read-only. The original sample folders and legacy evaluation artifacts were retained in place. The migration utility passed a checksum-conflict, retry, and repeat-run check using temporary fixture files.
