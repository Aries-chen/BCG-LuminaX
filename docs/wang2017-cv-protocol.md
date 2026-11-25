# Wang 2017 five-fold protocol

The fixed split uses the 125 eligible Wang v3 recordings, with one unique record per patient. The two nested splits use stratified five-fold with seed 42 and the four reference REI bins. Each fold has 80 training, 20 validation, and 25 outer-test subjects. The five outer tests cover the cohort once. REI uses artifact-masked recording hours. [Frozen split](../splits/wang2017_5fold_seed42.json) · [Dataset analysis](wang2017-bcg-data-report.md)

The singleton patient groups use `StratifiedKFold`. The split builder uses `StratifiedGroupKFold` when a patient has multiple records. The locked clinical bins contain 13, 40, 43 and 29 patients. Each outer test is within one patient per bin of the pooled target proportions. The outer-test mean REI ranges from 19.98 to 25.37 events per valid recording hour, with an omnibus split-balance ANOVA p-value of 0.811.

The waveform-only quality rules mark non-finite seconds, flat seconds, seconds with at least 25% samples at 0 or 4095, and runs of at least 30 seconds with peak-to-peak at most two. The last rule flags suspected off-bed periods. It triggered zero times in the 125 records. The resulting valid duration is 3,894,371 seconds, with 23,179 reference events entirely within valid spans. The original short annotations remain eligible. A sensitivity analysis uses references of at least ten seconds.

## Architecture and optimization audit

The `seqProj` head emits one event logit per second. `Pos` and `Len` define the reference interval for binary segmentation. There is no learned duration output. The CNN receptive fields are 2,331 samples at 50 Hz and 7,531 samples at 250 Hz, or 46.62 and 30.12 seconds. Bidirectional RoFormer attention supplies theoretical context across the 120-second window. The attention mask excludes padded and invalid seconds.

Each fold estimates the two branch P1/P99 bounds, median, and IQR from up to one million deterministically sampled valid training waveform values per branch. It computes class weight from all valid training seconds. Validation and outer-test data use that fold's frozen transform. The continuous cache preserves recording boundaries and uses 50/250 Hz inputs. Training samples 120-second windows with probabilities proportional to valid recording time and augments them with amplitude scaling and low-variance Gaussian noise. Validation and inference fuse 120-second windows at a 60-second stride with positive triangular weights.

Training uses batch size 64, AdamW at `3e-4`, five warmup epochs, cosine decay to `1e-6`, weight decay `1e-2`, and gradient clipping at 1.0. Forward uses BF16 CUDA autocast and the loss uses FP32 masked weighted BCE plus Soft Dice. Checkpoint selection ranks validation macro AP, default REI MAE, then epoch. Only the training and validation roles enter the optimizer and checkpoint selection. A fold uses the `1e-4` fallback once if its early divergence guard triggers.

## Launch and freeze

Submit from the project root with `ahi/run_wang2017_cv.pbs` and use one shared scratch `CV_ROOT`. Set `CV_STAGE=smoke` and `CV_FOLD=0` first. After smoke passes, run `CV_STAGE=train` with fold indices 0 through 4. The same launcher supports `CV_STAGE=tune` and `CV_STAGE=test`. A test run requires five validation-tuned configurations with verified checkpoint digests.

```bash
qsub -v CV_STAGE=train,CV_FOLD=0,CV_ROOT=/scratch/s5415436/BCG/runs/wang2017-5fold-seed42 \
  -l 'select=1:ncpus=8:ngpus=1:mem=32gb,walltime=04:00:00' ahi/run_wang2017_cv.pbs
```

The validation grid evaluates 156 combinations per fold. Selection minimizes patient REI MAE among
configurations with event F1 at least 90% of the fold maximum at IoU 0.3. A frozen record binds the
selected settings to the split, validation identities, and SHA-256 of `best.pt`.

Once the five outer-test jobs finish, aggregate the saved predictions from `graphiti_env`.

```bash
python -B ahi/report_wang2017_cv.py --root /scratch/s5415436/BCG/runs/wang2017-5fold-seed42
```

The completed run used PBS array `144256[].cadmin` for validation calibration and `144257[].cadmin`
for outer-test inference. Every subjob exited with status 0. The pooled evaluation covers all 125
patients exactly once, with 1,081.77 hours of valid coverage. [Training summary](../reports/wang2017_5fold_training.md) · [OOF evaluation](../reports/wang2017_5fold_oof_metrics.md)
