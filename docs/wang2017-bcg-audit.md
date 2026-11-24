# Wang 2017 BCG dataset audit

The local Figshare v3 archive contains 131 BCG waveforms and 136 event files. Each waveform has a same-ID event file. Five waveform downloads return HTTP 404. The 131 available waveforms contain 128 distinct decoded arrays. [Figshare v3 file manifest](https://api.figshare.com/v2/articles/4868102/versions/3) · [Wang et al. 2017](https://doi.org/10.1371/journal.pone.0175351)

## Archive and source verification

The archive at `/scratch/s5415436/data/BCG/wang2017_bcg_v3_partial.zip` passed `unzip -tq`. All 268 members matched the byte sizes and MD5 hashes in the v3 manifest, which lists 273 files. The `readme.txt` identifies the waveform sample rate as 100 Hz. Archive SHA-256 is `3d34a879eda48a5e4bb74f6a8bd431bbf6c02b98f4f5597a70f8ba586bdb5a2d`.

| Missing waveform | Figshare file ID |
| --- | ---: |
| `1058_012128.mat` | 8067236 |
| `1062_225704.mat` | 8067251 |
| `1084_204915.mat` | 8067320 |
| `1098_200008.mat` | 8067362 |
| `1123_215248.mat` | 8067437 |

All five corresponding `SASEvents` files are in the archive. A direct range request to each official waveform URL returned 404 on 2026-11-25. The same request against `1001_231215.mat` returned 206.

## Signal and label findings

All available waveforms loaded as finite, nonconstant vectors. SHA-256 hashes of the decoded NumPy arrays identified the following identical waveforms. The internal MATLAB variable's start-time suffix matches the filename in the first column and differs from the filename in the second. The three pairs should stay together or be resolved before splitting by subject.

| Filename matches embedded start time | Filename differs from embedded start time |
| --- | --- |
| `1001_231215.mat` | `1002_232917.mat` |
| `1013_232006.mat` | `1014_231208.mat` |
| `1016_224010.mat` | `1017_000557.mat` |

The 136 event files are readable. `SASEvents1013.mat` inserts a Chinese event-type field before `Pos` and `Len`. It contains 149 rows, including 97 central, 48 obstructive, and 4 hypopnea entries. The other event files use the five columns described in the README. After accounting for the extra field, all 131 pairs have nonnegative positions, positive lengths, and `Pos + Len` within the 100 Hz waveform duration under the `Pos`-based time origin.

The offset between the event wall clock and `filename start + Pos` is constant within each record but differs across records. Mapping `Time` onto the filename start puts an event beyond the waveform for IDs `1018`, `1086`, and `1105`. Observed waveform lengths range from 0.85 to 12.15 hours, with three recordings shorter than six hours (`1136`, `1021`, `1018`). These timing cases need review before whole-night scoring or strict boundary evaluation.

## Implications for the first BCG run

The archive supports a respiratory-event smoke run with the existing SleepNetZero backbone after adding an adapter for the single-channel 100 Hz BCG and `SASEvents` intervals. The release provides event intervals. Subject-level reference AHI, scored sleep duration, per-epoch stages, and severity classes would require another source. The paper reports severity groups and excludes wake in its AHI evaluation. [Wang et al. 2017](https://doi.org/10.1371/journal.pone.0175351)
