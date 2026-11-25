import copy
import json
import unittest

import numpy as np
from scipy.stats import f_oneway
import torch

from train_wang2017_cv import Packed, segmentation_loss, whole_record
from wang2017_cv_data import SPLIT, quality_mask, validate_split
from wang2017_cv_events import decode, event_counts, smooth_valid


class Wang2017CrossValidationTests(unittest.TestCase):
    def test_locked_split_and_leakage_guard(self):
        manifest = json.loads(SPLIT.read_text())
        validate_split(manifest)
        self.assertEqual(len(manifest["subjects"]), 125)
        self.assertEqual(manifest["clinical_bins"], [13, 40, 43, 29])
        all_bins = np.array(manifest["clinical_bins"])
        test_rei = []
        for fold in manifest["folds"]:
            self.assertEqual([len(fold[role]["subjects"]) for role in ("train", "val", "test")], [80, 20, 25])
            ids = fold["test"]["subjects"]
            test_rei.append([manifest["subjects"][sid]["rei"] for sid in ids])
            actual = np.bincount([manifest["subjects"][sid]["clinical_bin"] for sid in ids], minlength=4)
            np.testing.assert_array_equal(fold["test"]["clinical_bins"], actual)
            self.assertTrue(np.all(np.abs(actual - all_bins / 5) < 1))
        self.assertGreater(f_oneway(*test_rei).pvalue, .05)
        bad = copy.deepcopy(manifest)
        bad["folds"][0]["train"]["subjects"].append(bad["folds"][0]["test"]["subjects"][0])
        with self.assertRaisesRegex(ValueError, "subject overlap"):
            validate_split(bad)

    def test_quality_mask(self):
        waveform = np.tile(np.arange(100) + 2000., (38, 1))
        waveform[2, :25] = 4095
        waveform[3, 0] = np.nan
        waveform[4, :] = 2048
        waveform[5:35, :] = np.arange(100) % 3 + 1000
        valid, counts = quality_mask(waveform.flatten())
        self.assertEqual(counts, dict(missing=1, flat=1, saturated=1, suspected_off_bed=31))
        self.assertEqual(np.flatnonzero(valid).tolist(), [0, 1, 35, 36, 37])

    def test_tail_coverage_and_masked_loss(self):
        class Constant(torch.nn.Module):
            def forward(self, valid_mask, **kwargs):
                return {"ahi_event": torch.zeros_like(valid_mask, dtype=torch.float32)}

        for length in (1, 119, 120, 121, 181):
            valid = np.ones(length, dtype=bool)
            if length > 1:
                valid[1] = False
            feat = {"heartbreath": np.zeros((length, 50), np.float32), "snore": np.zeros((length, 250), np.float32)}
            entries = [dict(record_id="r", subject_id="s", start=0, stop=length, events=[])]
            params = {k: dict(lo=-1, hi=1, median=0, iqr=1) for k in feat}
            data = Packed(feat, np.zeros(length, dtype=np.uint8), valid, entries, params, torch.device("cpu"))
            records, _ = whole_record(Constant(), data)
            np.testing.assert_array_equal(records[0].probabilities, valid * 0.5)
        logits = torch.tensor([0., 0., 0.], requires_grad=True)
        masked = segmentation_loss(logits, torch.tensor([1., 0., 1.]), torch.tensor([True, True, False]), torch.tensor(2.0))
        masked.backward()
        self.assertEqual(logits.grad[-1].item(), 0.)

    def test_discontinuous_postprocessing_and_one_to_one(self):
        valid = np.ones(42, dtype=bool)
        valid[20] = False
        probabilities = np.ones(42, dtype=np.float32)
        smooth = smooth_valid(probabilities, valid, 7)
        events = decode(smooth, valid, 0.5, 6, 10)
        np.testing.assert_array_equal(events, [[0, 20], [21, 42]])
        self.assertEqual(event_counts(np.array([[0, 10], [10, 20]]), np.array([[0, 20]]), 0.5), (1, 0, 1))
        self.assertEqual(decode(np.ones(10), np.ones(10, bool), 0.5, 2, 10).tolist(), [[0, 10]])


if __name__ == "__main__":
    unittest.main()
