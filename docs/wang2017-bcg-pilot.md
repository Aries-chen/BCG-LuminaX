# Wang 2017 BCG smoke and 32-patient training pilot

The two-record CUDA smoke passed. The 32-patient pilot met both pre-specified convergence gates. The held-out four-patient test AUPRC was **0.5093** at a positive-second fraction of **0.2526**. These results cover the sampled 120-second windows in one patient-wise split and one seed.

## Patient split and input

The pilot selected eight eligible records in each quartile of per-record marked-event seconds divided by recording duration. Each quartile contributed six training, one validation, and one test patient. The two smoke IDs `1001` and `1013` were assigned to training first. Remaining assignments used NumPy `default_rng(42)` on IDs sorted by event-second fraction and then by ID. The fixed split and saved window metadata have disjoint training, validation, and test IDs. [Eligibility audit](wang2017-bcg-audit.md) · [Fixed split](wang2017-bcg-pilot-split.csv)

Each training patient supplied eight mixed-label and eight all-negative windows, for 384 windows. Each validation and test patient supplied 48 uniformly spaced windows, for 192 in each split. The model predicts one binary event score per second using the `Pos` and `Len` intervals. The input uses the recorded single-channel 100 Hz BCG, filtered to 0.1–20 Hz and 30–45 Hz, standardized per record, and resampled to 50 Hz and 250 Hz for the two existing SleepNetZero branches. The 250 Hz tensor contains the filtered BCG band. The network uses two 64-dimensional RoFormer layers, the existing ResNet extractors, and the `seqProj` event head.

## GPU smoke

PBS job `144124.cadmin` completed on an A100 80 GB in 13 seconds. The 120-second windows started at 2760 s for `1001` and 120 s for `1013`, with 10 and 28 positive seconds. The batch produced `[2, 120]` logits and loss `0.65216`. Gradients were finite, the optimizer updated the event head, and the reloaded checkpoint reproduced the logits.

## Convergence and held-out result

PBS job `144125.cadmin` completed on an A100 80 GB in 27 seconds. It used FP32, batch size 8, Adam with learning rate `1e-4`, and seed `42` in the existing `graphiti_env`. The pilot evaluates unweighted binary cross-entropy and computes AUPRC as average precision over scored seconds.

| Epoch | Training loss | Validation loss | Validation AUPRC |
| ---: | ---: | ---: | ---: |
| 0 | — | 0.89329 | 0.20655 |
| 1 | 0.58135 | 0.53166 | 0.20379 |
| 2 | 0.49153 | 0.49975 | 0.25363 |
| 3 | 0.47236 | 0.47639 | 0.32965 |
| 4 | 0.45577 | 0.46514 | 0.37293 |
| 5 | 0.45217 | 0.46309 | 0.37166 |
| 6 | 0.44505 | 0.44243 | 0.43364 |
| 7 | 0.43571 | 0.44043 | 0.42478 |
| 8 | **0.43102** | **0.42695** | **0.47107** |

Epoch 8 had the highest validation AUPRC. Its training loss was 25.9% below epoch 1 and cleared the `0.52321` cutoff. Its validation AUPRC exceeded the `0.22655` cutoff by `0.24452`. Validation prevalence was `0.20239`.

The selected checkpoint was evaluated once on test patients `1016`, `1064`, `1100`, and `1113`. Test AUPRC was `0.50929`, AUROC was `0.72169`, positive-second fraction was `0.25256`, and BCE was `0.52775`. Recomputing the metrics from the saved logits and labels reproduced the summary.

## Reproduction and run artifacts

The [PBS commands](../README.md#wang-2017-bcg-event-detection) submit the smoke followed by the pilot. Outputs, including the metrics CSV, split snapshot, checkpoint, and predictions, are under `/scratch/s5415436/BCG/runs/wang2017-bcg-pilot-144125.cadmin/`. The smoke output is under `/scratch/s5415436/BCG/runs/wang2017-bcg-smoke-144124.cadmin/`.
