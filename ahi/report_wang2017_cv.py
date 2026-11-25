"""Aggregate the frozen Wang 2017 OOF predictions into reports and research figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve

from train_wang2017_cv import frozen_folds
from wang2017_cv_data import ROOT, SEED, SPLIT, file_hash, get_records, spans, validate_split, write_json
from wang2017_cv_events import RecordScores, divide, event_metrics, patient_rows

CLASSES = ("<5", "5–<15", "15–<30", "≥30")
CUTS = (5, 15, 30)
COLORS = ("#287C77", "#3B6DAB", "#AB6A23", "#9B415B")
BOOTSTRAPS = 2000


def _write_csv(path, rows, *, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _icc(reference, predicted):
    # Two-way random effects, absolute agreement, single measurement.
    scores = np.column_stack((reference, predicted))
    n, k = scores.shape
    grand = scores.mean()
    subjects, raters = scores.mean(axis=1), scores.mean(axis=0)
    row_ms = k * np.sum((subjects - grand)**2) / (n - 1)
    col_ms = n * np.sum((raters - grand)**2) / (k - 1)
    err_ms = np.sum((scores - subjects[:, None] - raters + grand)**2) / ((n - 1) * (k - 1))
    return divide(row_ms - err_ms, row_ms + (k - 1) * err_ms + k * (col_ms - err_ms) / n)


def _cutpoint(reference, predicted, cutoff):
    y, x = reference >= cutoff, predicted >= cutoff
    tp = int((y & x).sum())
    tn = int((~y & ~x).sum())
    fp = int((~y & x).sum())
    fn = int((y & ~x).sum())
    return dict(sensitivity=divide(tp, tp + fn), specificity=divide(tn, tn + fp),
                ppv=divide(tp, tp + fp), npv=divide(tn, tn + fn), f1=divide(2 * tp, 2 * tp + fn + fp),
                auroc=float(roc_auc_score(y, predicted)) if y.any() and not y.all() else None,
                tp=tp, tn=tn, fp=fp, fn=fn)


def _bootstrap(reference, predicted):
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(len(reference), size=(BOOTSTRAPS, len(reference)))
    ref, pred = reference[sampled], predicted[sampled]
    ranks = rankdata(pred, axis=1)
    intervals = {}
    for cutoff in CUTS:
        positive, flagged = ref >= cutoff, pred >= cutoff
        tp, fn = (positive & flagged).sum(1), (positive & ~flagged).sum(1)
        tn, fp = (~positive & ~flagged).sum(1), (~positive & flagged).sum(1)
        n_pos, n_neg = positive.sum(1), (~positive).sum(1)
        wins = (ranks * positive).sum(1) - n_pos * (n_pos + 1) / 2
        pairs = {"sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp),
                 "ppv": (tp, tp + fp), "npv": (tn, tn + fn),
                 "f1": (2 * tp, 2 * tp + fp + fn), "auroc": (wins, n_pos * n_neg)}
        limits = {}
        for key, (num, den) in pairs.items():
            usable = den > 0
            q = np.quantile(num[usable] / den[usable], [.025, .975]).tolist() if usable.any() else None
            limits[key] = {"ci95": q, "usable_replicates": int(usable.sum())}
        intervals[str(cutoff)] = limits
    return intervals


def summarize(rows, bootstrap=False):
    reference = np.array([x["reference_rei"] for x in rows])
    predicted = np.array([x["predicted_rei"] for x in rows])
    difference = predicted - reference
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    correlations = dict(pearson_r=None, pearson_p=None, pearson_ci95=None, spearman_rho=None)
    if np.ptp(reference) > 0 and np.ptp(predicted) > 0:
        result = pearsonr(reference, predicted)
        correlations = dict(pearson_r=float(result.statistic), pearson_p=float(result.pvalue),
                            pearson_ci95=list(map(float, result.confidence_interval())),
                            spearman_rho=float(spearmanr(reference, predicted).statistic))
    actual_bin, predicted_bin = (np.digitize(x, CUTS) for x in (reference, predicted))
    cm = confusion_matrix(actual_bin, predicted_bin, labels=range(4))
    class_f1 = [divide(2 * cm[i, i], cm[i, :].sum() + cm[:, i].sum()) for i in range(4)]
    kappa = {}
    for mode in ("linear", "quadratic"):
        kappa[mode] = (float(cohen_kappa_score(actual_bin, predicted_bin, labels=list(range(4)), weights=mode))
                       if len(set(actual_bin) | set(predicted_bin)) > 1 else None)
    metrics = {
        "patients": len(rows),
        "events": {str(iou): event_metrics(np.asarray([r["counts"][iou] for r in rows]).sum(axis=0)) for iou in (0.3, 0.5)},
        "continuous": dict(**correlations, icc_2_1=_icc(reference, predicted), mae=float(np.abs(difference).mean()),
                           rmse=float(np.sqrt(np.mean(difference**2))), mean_bias=bias,
                           loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd),
        "classification": dict(accuracy=float(np.mean(actual_bin == predicted_bin)),
                               macro_f1=float(np.mean([x for x in class_f1 if x is not None])),
                               linear_kappa=kappa["linear"], quadratic_kappa=kappa["quadratic"],
                               confusion_counts=cm.tolist(),
                               row_percent=[[divide(100 * c, row.sum()) for c in row] for row in cm]),
        "cutpoints": {str(cutoff): _cutpoint(reference, predicted, cutoff) for cutoff in CUTS},
    }
    if bootstrap:
        metrics["cutpoint_patient_bootstrap"] = _bootstrap(reference, predicted)
    return metrics


def scalar_metrics(metrics):
    flat = {}
    for group, mapping in metrics.items():
        if group not in ("events", "continuous", "classification", "cutpoints"):
            continue
        for name, value in mapping.items():
            if isinstance(value, dict):
                flat.update({f"{group}/{name}/{key}": v for key, v in value.items()})
            elif value is None or isinstance(value, (float, int)):
                flat[f"{group}/{name}"] = value
    return flat


def fold_stability(fold_metrics):
    flat = [scalar_metrics(m) for m in fold_metrics]
    result = {}
    for key in flat[0]:
        values = [m.get(key) for m in flat if m.get(key) is not None]
        result[key] = {"mean": float(np.mean(values)) if values else None,
                       "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "n_folds": len(values)}
    return result


def _save_figure(figure, folder, name):
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_figures(rows, examples, metrics, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    ref = np.array([r["reference_rei"] for r in rows])
    pred = np.array([r["predicted_rei"] for r in rows])
    classes = np.digitize(ref, CUTS)
    c = metrics["continuous"]
    fig, ax = plt.subplots(figsize=(6.3, 5.3), layout="constrained")
    hi = max(ref.max(), pred.max()) * 1.04
    line = np.array([0., hi])
    ax.plot(line, line, "--", color="#777777", lw=1, label="Identity")
    ax.plot(line, np.polyval(np.polyfit(ref, pred, 1), line), color="#202020", lw=1, label="Linear fit")
    for i, (color, name) in enumerate(zip(COLORS, CLASSES)):
        valid = classes == i
        ax.scatter(ref[valid], pred[valid], s=28, color=color, alpha=.85, label=name)
    for cutoff in CUTS:
        ax.axvline(cutoff, color="#dddddd", lw=.7, zorder=0)
        ax.axhline(cutoff, color="#dddddd", lw=.7, zorder=0)
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="Reference REI (events / valid recording hour)",
           ylabel="Predicted REI (events / valid recording hour)", title="Patient-level REI | out-of-fold")
    ax.text(.04, .96, f"N={len(ref)}\nPearson r={_fmt(c['pearson_r'])}\nICC={_fmt(c['icc_2_1'])}\nMAE={c['mae']:.2f}",
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    _save_figure(fig, folder, "fig3_rei_scatter")

    fig, ax = plt.subplots(figsize=(7.1, 4.5), layout="constrained")
    ax.scatter((ref + pred) / 2, pred - ref, c=np.asarray(COLORS)[classes], s=27, alpha=.85)
    for key, text, line in (("mean_bias", "Bias", "-"), ("loa_low", "Lower LoA", "--"), ("loa_high", "Upper LoA", "--")):
        y = c[key]
        ax.axhline(y, color="#25374A", linestyle=line, linewidth=1)
        ax.annotate(f"{text} {y:.2f}", xy=(1., y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
    ax.set(xlabel="Mean of reference and predicted REI", ylabel="Predicted − reference REI",
           title=f"REI agreement | Bland–Altman | N={len(ref)}")
    _save_figure(fig, folder, "fig4_bland_altman")

    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout="constrained")
    cls = metrics["classification"]
    count, values = np.array(cls["confusion_counts"]), np.array(cls["row_percent"], dtype=float)
    edges = np.arange(5) - .5
    mapped = ax.pcolormesh(edges, edges, values, cmap="Blues", vmin=0, vmax=100)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for i, j in np.ndindex(count.shape):
        value = values[i, j]
        text = f"{count[i,j]}\n{value:.1f}%" if np.isfinite(value) else f"{count[i,j]}\nNA"
        ax.text(j, i, text, ha="center", va="center", color="white" if value > 60 else "#223649")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted REI bin", ylabel="Reference REI bin",
           title=f"OOF REI bins | quadratic κ={cls['quadratic_kappa']:.3f}")
    colorbar = fig.colorbar(mapped, ax=ax, label="Percent within reference bin")
    colorbar.solids.set_rasterized(False)
    _save_figure(fig, folder, "supp_rei_confusion")

    fig, ax = plt.subplots(figsize=(5.7, 4.75), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", lw=.8, color="#777777")
    for cutoff, color in zip(CUTS, COLORS[1:]):
        fpr, tpr, _ = roc_curve(ref >= cutoff, pred)
        key = str(cutoff)
        values = metrics["cutpoints"][key]
        lower, upper = metrics["cutpoint_patient_bootstrap"][key]["auroc"]["ci95"]
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"≥{cutoff}: {values['auroc']:.3f} (95% CI {lower:.3f}–{upper:.3f})")
        ax.scatter(1 - values["specificity"], values["sensitivity"], color=color, s=20)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate",
           title="OOF REI cutpoints | patient bootstrap")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _save_figure(fig, folder, "supp_rei_cutpoint_roc")

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    outer = fig.add_gridspec(len(examples), 1)
    for i, (subject, data, patient) in enumerate(examples):
        gs = outer[i].subgridspec(3, 1, hspace=.01, height_ratios=(1, 1, .62))
        a = fig.add_subplot(gs[0, 0])
        b = fig.add_subplot(gs[1, 0], sharex=a)
        d = fig.add_subplot(gs[2, 0], sharex=a)
        time_hours = np.arange(len(data["probabilities"])) / 3600
        a.plot(time_hours, np.where(data["valid"], data["envelope"], np.nan), color=COLORS[0], lw=.55)
        b.plot(time_hours, np.where(data["valid"], data["probabilities"], np.nan), color=COLORS[1], lw=.55)
        for tick, row, color in ((1, "reference", COLORS[2]), (0, "predicted", COLORS[3])):
            intervals = data[row] / 3600
            d.broken_barh([(start, stop - start) for start, stop in intervals], (tick - .29, .58), facecolors=color)
        for start, stop in spans(~data["valid"]) / 3600:
            d.axvspan(start, stop, color="#dbdce0", lw=0, zorder=0)
        a.set(ylabel="BCG RMS", title=f"Held-out subject {subject} | reference {patient['reference_rei']:.1f} | predicted {patient['predicted_rei']:.1f}")
        a.title.set_horizontalalignment("left")
        a.title.set_position((0., 1.))
        b.set(ylim=(0, 1), ylabel="Probability")
        d.set(yticks=(0, 1), yticklabels=("Pred.", "Ref."), ylim=(-.5, 1.5),
              xlabel="Hours from recording start", xlim=(0, len(data["probabilities"]) / 3600))
        a.tick_params(labelbottom=False)
        b.tick_params(labelbottom=False)
    _save_figure(fig, folder, "supp_whole_record_examples")


def _load_outputs(root, manifest, frozen):
    folds, event_table, recorded = [], [], {}
    for i in range(5):
        out = root / f"fold-{i}"
        complete = json.loads((out / "test_complete.json").read_text())
        allowed = get_records(manifest, i, "test", allow_test=True)
        if complete["records"] != [r["record_id"] for r in allowed] or complete["checkpoint_sha256"] != frozen[i]["checkpoint_sha256"]:
            raise ValueError(f"fold {i}: OOF completion marker differs from freeze")
        recordings, intervals = [], []
        for row in allowed:
            rid, sid = row["record_id"], row["subject_id"]
            path = out / "outer_test" / f"{rid}.npz"
            if rid in recorded:
                raise ValueError(f"{rid}: repeated OOF record")
            recorded[rid] = path
            with np.load(path) as item:
                if item["subject_id"].item() != sid or item["record_id"].item() != rid or item["checkpoint_sha256"].item() != complete["checkpoint_sha256"]:
                    raise ValueError(f"{rid}: inconsistent OOF identity")
                r = RecordScores(sid, rid, item["probabilities"], item["valid"], item["labels"], item["reference"])
                expected = (row["seconds"],)
                if (any(x.shape != expected for x in (r.probabilities, r.valid, r.labels))
                        or int(r.valid.sum()) != row["valid_seconds"] or len(r.references) != row["valid_events"]):
                    raise ValueError(f"{rid}: OOF coverage differs from lock")
                if (not np.isfinite(r.probabilities).all() or np.any((r.probabilities < 0) | (r.probabilities > 1))
                        or np.any(r.probabilities[~r.valid])):
                    raise ValueError(f"{rid}: invalid stitched probabilities")
                predicted = item["predicted"]
                start, stop = predicted.T
                if (np.any(start < 0) or np.any(stop > row["seconds"])
                        or np.any(stop - start < frozen[i]["postprocessing"]["minimum"])
                        or np.any(start[1:] < stop[:-1])):
                    raise ValueError(f"{rid}: invalid predicted event intervals")
                invalid_prefix = np.r_[0, np.cumsum(~r.valid)]
                if np.any(invalid_prefix[start] != invalid_prefix[stop]):
                    raise ValueError(f"{rid}: predicted event crosses invalid recording time")
                recordings.append(r)
                intervals.append(predicted)
                event_table.extend(dict(fold=i, subject_id=sid, record_id=rid, start_s=a, end_s=b)
                                   for a, b in predicted.tolist())
        primary = patient_rows(recordings, intervals)
        sensitivity = patient_rows(recordings, intervals, min_reference=10)
        for row in primary + sensitivity:
            row["fold"] = i
        folds.append((primary, sensitivity))
    if set(recorded) != set(manifest["records"]):
        raise ValueError("unexpected OOF record coverage")
    return folds, event_table, recorded


def _fmt(value, spec=".4f"):
    return "NA" if value is None else format(value, spec)


def _markdown(result):
    plus = result["fold_stability"]
    pooled = scalar_metrics(result["pooled_oof"])
    n = result["pooled_oof"]["patients"]
    lines = ["# Wang 2017 five-fold OOF evaluation", "",
             f"The pooled out-of-fold set contains **{n} patients**. Each patient was scored with the checkpoint",
             "and postprocessing selected on its fold's development data. REI uses valid recording hours (TRT).",
             "The four bins and screening cutpoints are defined on reference REI in this cohort.", "",
             "| Metric | Pooled OOF | Fold mean ± SD | Folds |", "| --- | ---: | ---: | ---: |"]
    for key, value in pooled.items():
        if key.rsplit("/", 1)[-1] in ("tp", "fp", "tn", "fn"):
            continue
        fold = plus[key]
        spec = ".3g" if key == "continuous/pearson_p" else ".4f"
        lines.append(f"| {key} | {_fmt(value, spec)} | {_fmt(fold['mean'], spec)} ± {_fmt(fold['sd'], spec)} | {fold['n_folds']} |")
    p = result["pooled_oof"]["continuous"]
    ci = p["pearson_ci95"]
    lines.extend(["", f"Pearson r 95% CI is [{ci[0]:.4f}, {ci[1]:.4f}]." if ci else "Pearson CI is NA.",
                  "Fold means and SDs use the folds with a defined value, as counted in the final column.", "",
                  "## Screening cutpoints", "",
                  "The 95% confidence intervals use 2,000 patient bootstrap replicates.", "",
                  "| Reference REI cutpoint | Sensitivity (95% CI) | Specificity (95% CI) | AUROC (95% CI) |",
                  "| --- | ---: | ---: | ---: |"])
    for cutoff, metrics in result["pooled_oof"]["cutpoints"].items():
        bootstrap = result["pooled_oof"]["cutpoint_patient_bootstrap"][cutoff]
        cells = []
        for key in ("sensitivity", "specificity", "auroc"):
            ci = bootstrap[key]["ci95"]
            interval = f"{_fmt(ci[0])}–{_fmt(ci[1])}" if ci else "NA"
            cells.append(f"{_fmt(metrics[key])} ({interval})")
        lines.append(f"| ≥{cutoff} | {' | '.join(cells)} |")
    lines.extend(["", "## Frozen postprocessing", "",
                  "| Fold | Smoothing (s) | Threshold | Merge gap (s) | Minimum (s) | Val REI MAE | Val F1 at IoU 0.3 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for fold in result["frozen_folds"]:
        p = fold["postprocessing"]
        lines.append(f"| {fold['fold']} | {p['smooth']} | {p['threshold']:.2f} | {p['gap']} | {p['minimum']} | "
                     f"{p['rei_mae']:.4f} | {p['event_f1']:.4f} |")
    lines.extend(["", "## Reference events at least 10 seconds", "",
                  "The sensitivity analysis uses the same predictions, quality masks and frozen postprocessing.", ""])
    sens = result["reference_min_10s_sensitivity"]["pooled_oof"]
    lines.extend([f"- Event F1 at IoU 0.3 is {_fmt(sens['events']['0.3']['f1'])}.",
                  f"- Event F1 at IoU 0.5 is {_fmt(sens['events']['0.5']['f1'])}.",
                  f"- Patient REI MAE is {_fmt(sens['continuous']['mae'])} events per valid recording hour.", "",
                  "## Files", "",
                  f"The run outputs are under `{result['run_directory']}`. Each figure has a vector PDF and a 300 DPI PNG.",
                  "",
                  "- `figures/fig3_rei_scatter` shows patient-level REI agreement.",
                  "- `figures/fig4_bland_altman` shows bias and limits of agreement.",
                  "- `figures/supp_rei_confusion` shows the clinical bin counts and row proportions.",
                  "- `figures/supp_rei_cutpoint_roc` shows ROC curves and patient-bootstrap AUROC intervals.",
                  "- `figures/supp_whole_record_examples` shows the held-out normal, moderate and severe records",
                  "  nearest the median reference REI within their bin. Gray marks excluded recording time.",
                  "",
                  "The revised figure set omits a numbered Fig. 5. The respiratory event annotations support the",
                  "REI analyses above. TST, SOL, WASO, sleep efficiency, and REM latency require sleep-stage",
                  "scores, which are unavailable in the Wang 2017 release.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    meta = json.loads(args.split.read_text())
    validate_split(meta)
    split_hash = file_hash(args.split)
    frozen = frozen_folds(args.root, meta, split_hash)
    folds, events, file_index = _load_outputs(args.root, meta, frozen)
    pooled = {}
    for name, j in (("primary", 0), ("reference_min_10s_sensitivity", 1)):
        rows = [r for fold in folds for r in fold[j]]
        if len(rows) != len(meta["subjects"]) or len({r["subject_id"] for r in rows}) != len(rows):
            raise ValueError("unexpected OOF patient coverage")
        if j == 0:
            primary = sorted(rows, key=lambda r: r["subject_id"])
        by_fold = [summarize(fold[j]) for fold in folds]
        pooled[name] = dict(pooled_oof=summarize(rows, bootstrap=(j == 0)), folds=by_fold,
                            fold_stability=fold_stability(by_fold))
    result = dict(run_directory=str(args.root), split_sha256=split_hash, frozen_folds=frozen,
                  **pooled["primary"], reference_min_10s_sensitivity=pooled["reference_min_10s_sensitivity"])
    examples = []
    for selected_class in (0, 2, 3):
        eligible = [r for r in primary if np.digitize(r["reference_rei"], CUTS) == selected_class]
        median = np.median([r["reference_rei"] for r in eligible])
        row = min(eligible, key=lambda r: (abs(r["reference_rei"] - median), r["subject_id"]))
        rid = next(key for key, value in meta["records"].items() if value["subject_id"] == row["subject_id"])
        with np.load(file_index[rid]) as archive:
            examples.append((row["subject_id"], dict(archive), row))
    result["example_subjects"] = [r[0] for r in examples]
    make_figures(primary, examples, result["pooled_oof"], args.root / "figures")
    _write_csv(args.root / "oof_patients.csv", [
        {**{k: r[k] for k in ("subject_id", "fold", "seconds", "gt", "pred", "reference_rei", "predicted_rei")},
         **{f"{part}_iou_{iou}": val for iou, cnt in r["counts"].items() for part, val in zip(("tp", "fp", "fn"), cnt)}}
        for r in primary])
    _write_csv(args.root / "oof_predicted_events.csv", events,
               fields=["fold", "subject_id", "record_id", "start_s", "end_s"])
    args.reports.mkdir(parents=True, exist_ok=True)
    write_json(args.reports / "wang2017_5fold_oof_metrics.json", result)
    (args.reports / "wang2017_5fold_oof_metrics.md").write_text(_markdown(result))
    print(json.dumps(result["pooled_oof"]["continuous"], allow_nan=False))


if __name__ == "__main__":
    main()
