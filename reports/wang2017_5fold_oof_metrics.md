# Wang 2017 five-fold OOF evaluation

The pooled out-of-fold set contains **125 patients**. Each patient was scored with the checkpoint
and postprocessing selected on its fold's development data. REI uses valid recording hours (TRT).
The four bins and screening cutpoints are defined on reference REI in this cohort.

| Metric | Pooled OOF | Fold mean ± SD | Folds |
| --- | ---: | ---: | ---: |
| events/0.3/precision | 0.5356 | 0.5388 ± 0.0822 | 5 |
| events/0.3/recall | 0.5175 | 0.5168 ± 0.0866 | 5 |
| events/0.3/f1 | 0.5264 | 0.5243 ± 0.0705 | 5 |
| events/0.5/precision | 0.4335 | 0.4353 ± 0.0806 | 5 |
| events/0.5/recall | 0.4189 | 0.4183 ± 0.0842 | 5 |
| events/0.5/f1 | 0.4260 | 0.4240 ± 0.0734 | 5 |
| continuous/pearson_r | 0.8152 | 0.8421 ± 0.0520 | 5 |
| continuous/pearson_p | 5.86e-31 | 1.76e-06 ± 3.42e-06 | 5 |
| continuous/spearman_rho | 0.6297 | 0.6366 ± 0.1408 | 5 |
| continuous/icc_2_1 | 0.8101 | 0.8108 ± 0.0746 | 5 |
| continuous/mae | 7.7511 | 7.7511 ± 1.7329 | 5 |
| continuous/rmse | 9.9881 | 9.8299 ± 1.9795 | 5 |
| continuous/mean_bias | -1.0792 | -1.0792 ± 3.8268 | 5 |
| continuous/loa_low | -20.6196 | -19.5191 ± 5.1173 | 5 |
| continuous/loa_high | 18.4613 | 17.3608 ± 4.7130 | 5 |
| classification/accuracy | 0.5120 | 0.5120 ± 0.0593 | 5 |
| classification/macro_f1 | 0.4745 | 0.4720 ± 0.1001 | 5 |
| classification/linear_kappa | 0.4344 | 0.4313 ± 0.1335 | 5 |
| classification/quadratic_kappa | 0.5728 | 0.5623 ± 0.2068 | 5 |
| cutpoints/5/sensitivity | 0.9554 | 0.9557 ± 0.0622 | 5 |
| cutpoints/5/specificity | 0.1538 | 0.1667 ± 0.2357 | 5 |
| cutpoints/5/ppv | 0.9068 | 0.9067 ± 0.0349 | 5 |
| cutpoints/5/npv | 0.2857 | 0.5000 ± 0.5774 | 4 |
| cutpoints/5/f1 | 0.9304 | 0.9298 ± 0.0403 | 5 |
| cutpoints/5/auroc | 0.7761 | 0.8076 ± 0.1221 | 5 |
| cutpoints/15/sensitivity | 0.6806 | 0.6810 ± 0.0598 | 5 |
| cutpoints/15/specificity | 0.6981 | 0.6982 ± 0.2258 | 5 |
| cutpoints/15/ppv | 0.7538 | 0.7799 ± 0.1397 | 5 |
| cutpoints/15/npv | 0.6167 | 0.6082 ± 0.0472 | 5 |
| cutpoints/15/f1 | 0.7153 | 0.7182 ± 0.0381 | 5 |
| cutpoints/15/auroc | 0.7804 | 0.7840 ± 0.0904 | 5 |
| cutpoints/30/sensitivity | 0.6552 | 0.6533 ± 0.2063 | 5 |
| cutpoints/30/specificity | 0.9583 | 0.9589 ± 0.0422 | 5 |
| cutpoints/30/ppv | 0.8261 | 0.8200 ± 0.1850 | 5 |
| cutpoints/30/npv | 0.9020 | 0.9041 ± 0.0533 | 5 |
| cutpoints/30/f1 | 0.7308 | 0.7174 ± 0.1906 | 5 |
| cutpoints/30/auroc | 0.8915 | 0.8831 ± 0.1297 | 5 |

Pearson r 95% CI is [0.7465, 0.8667].
Fold means and SDs use the folds with a defined value, as counted in the final column.

## Screening cutpoints

The 95% confidence intervals use 2,000 patient bootstrap replicates.

| Reference REI cutpoint | Sensitivity (95% CI) | Specificity (95% CI) | AUROC (95% CI) |
| --- | ---: | ---: | ---: |
| ≥5 | 0.9554 (0.9115–0.9910) | 0.1538 (0.0000–0.3846) | 0.7761 (0.6287–0.9075) |
| ≥15 | 0.6806 (0.5714–0.7888) | 0.6981 (0.5714–0.8163) | 0.7804 (0.7005–0.8577) |
| ≥30 | 0.6552 (0.4800–0.8215) | 0.9583 (0.9140–0.9901) | 0.8915 (0.8040–0.9577) |

## Frozen postprocessing

| Fold | Smoothing (s) | Threshold | Merge gap (s) | Minimum (s) | Val REI MAE | Val F1 at IoU 0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 0.65 | 6 | 10 | 7.0595 | 0.5422 |
| 1 | 3 | 0.65 | 2 | 10 | 8.3770 | 0.4461 |
| 2 | 1 | 0.70 | 6 | 10 | 8.4216 | 0.4936 |
| 3 | 1 | 0.75 | 2 | 10 | 5.7308 | 0.5275 |
| 4 | 7 | 0.70 | 4 | 10 | 7.0943 | 0.5472 |

## Reference events at least 10 seconds

The sensitivity analysis uses the same predictions, quality masks and frozen postprocessing.

- Event F1 at IoU 0.3 is 0.5329.
- Event F1 at IoU 0.5 is 0.4317.
- Patient REI MAE is 7.1711 events per valid recording hour.

## Files

The run outputs are under `/scratch/s5415436/BCG/runs/wang2017-5fold-seed42`. Each figure has a vector PDF and a 300 DPI PNG.

- `figures/fig3_rei_scatter` shows patient-level REI agreement.
- `figures/fig4_bland_altman` shows bias and limits of agreement.
- `figures/supp_rei_confusion` shows the clinical bin counts and row proportions.
- `figures/supp_rei_cutpoint_roc` shows ROC curves and patient-bootstrap AUROC intervals.
- `figures/supp_whole_record_examples` shows the held-out normal, moderate and severe records
  nearest the median reference REI within their bin. Gray marks excluded recording time.

The revised figure set omits a numbered Fig. 5. The respiratory event annotations support the
REI analyses above. TST, SOL, WASO, sleep efficiency, and REM latency require sleep-stage
scores, which are unavailable in the Wang 2017 release.
