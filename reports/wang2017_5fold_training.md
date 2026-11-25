# Wang 2017 five-fold training

PBS smoke job `144250.cadmin` and training array `144251[].cadmin` completed successfully
on NVIDIA A100 80 GB GPUs. The folds used batch size 64 and fresh initialization.
The five `best.pt` files loaded on CPU and matched the highest validation macro AP,
with lower default REI MAE and earlier epoch as tie breakers. Development cache record
IDs matched the split and were disjoint from each outer test.

## Fold 0 smoke

The two epochs took 13.24 s or less each after 67.8 s
of cache preparation. The peak CUDA allocation was 3.77 GiB.
PBS recorded 5.26 GiB of host memory for the smoke job.
The smoke restored `last.pt` after epoch 1 and ran the second epoch. The longest-epoch
estimate for all 80 epochs plus preparation was 18.8 minutes.

## Training and validation

| Fold | Epochs | Best epoch | Train loss, first → last | Val loss at best | Macro AP | Micro AP | Default REI MAE | Default event F1 | Runtime (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 26 | 14 | 0.9263 → 0.4896 | 0.7117 | 0.4399 | 0.5939 | 9.796 | 0.5270 | 4.54 |
| 1 | 22 | 10 | 0.9338 → 0.5051 | 0.7573 | 0.3278 | 0.5034 | 10.906 | 0.3863 | 4.97 |
| 2 | 24 | 12 | 0.9502 → 0.5169 | 0.7986 | 0.3669 | 0.5155 | 10.058 | 0.4618 | 5.36 |
| 3 | 24 | 12 | 0.9277 → 0.4892 | 0.7193 | 0.3301 | 0.5430 | 11.778 | 0.4738 | 5.33 |
| 4 | 27 | 15 | 0.9392 → 0.4715 | 0.7660 | 0.3972 | 0.6251 | 9.060 | 0.4932 | 6.11 |

Runtime covers preparation and training inside each job. Event F1 uses one-to-one matching at IoU 0.3.
The default postprocessing uses 1-second smoothing, a 0.50 probability threshold, a 2-second
merge gap, and a 10-second minimum event duration.

| Fold | Windows per epoch | Train-only positive weight | Initial learning rate | Cache and GPU preparation (s) | Peak CUDA allocation (GiB) | Stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 20,334 | 4.879 | 3e-04 | 1.2 | 3.77 | Validation patience |
| 1 | 20,726 | 4.815 | 3e-04 | 67.8 | 3.80 | Validation patience |
| 2 | 21,000 | 5.105 | 3e-04 | 68.0 | 3.85 | Validation patience |
| 3 | 20,917 | 4.741 | 3e-04 | 67.1 | 3.83 | Validation patience |
| 4 | 20,774 | 4.925 | 3e-04 | 72.1 | 3.82 | Validation patience |


Checkpoints, cache, and per-epoch CSVs are under `/scratch/s5415436/BCG/runs/wang2017-5fold-seed42`.
Validation REI uses hours of artifact-masked recording. Frozen calibration and OOF inference follow this gate.
