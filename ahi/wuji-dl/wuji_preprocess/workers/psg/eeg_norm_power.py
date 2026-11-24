import numpy as np
from numpy.typing import NDArray
import typing as t
import neurokit2 as nk
from scipy.integrate import simpson
from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker
from wuji.tools.sort import create_sort_key
import math

def get_eeg_features(
        eeg: NDArray,
        fs: t.Union[int, float],
        seg_duration:  t.Union[int, float] = 1,
        seg_overlap_ratio: t.Union[int, float] = 0.5,
        to_band: bool = False
) -> NDArray:
    """
    This function calculates the normalized power of an EEG signal.

    Parameters:
    eeg (numpy array): The EEG signal data.
    fs (int): The sampling frequency of the EEG signal.
    seg_duration (int, optional): The duration of each segment in seconds. Defaults to 1.
    seg_overlap_ratio (float, optional): The overlap ratio of each segment. Defaults to 0.5.

    Returns:
    norm_power (numpy array): The normalized power of the EEG signal. The dimensions are frequency by time.
    """
    if to_band:
        freq, time, power = nk.signal_timefrequency(
            eeg,
            window=seg_duration,
            overlap=fs * seg_duration * seg_overlap_ratio,
            sampling_rate=fs,
            show=False)
        power = bandpower_from_psd_ndarray(power.T, freq, relative=False)
        power = power / (np.sum(power, axis=0, keepdims=True) + 1e-8)
    else:
        freq, time, power = nk.signal_timefrequency(
            eeg,
            window=seg_duration,
            overlap=fs * seg_duration * seg_overlap_ratio,
            min_frequency=0.3,
            max_frequency=35,
            sampling_rate=fs,
            show=False)
        power = power / (np.sum(power, axis=0, keepdims=True) + 1e-8) # 正则化，要改！
    return power  # freq * time

def bandpower_from_psd_ndarray(
    psd,
    freqs,
    bands=[
        (0.5, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 16, "Sigma"),
        (16, 30, "Beta"),
        (30, 40, "Gamma"),
    ],
    relative=True,
):
    """Compute bandpowers in N-dimensional PSD.

    This is a NumPy-only implementation of the :py:func:`yasa.bandpower_from_psd` function,
    which supports 1-D arrays of shape (n_freqs), or N-dimensional arays (e.g. 2-D (n_chan,
    n_freqs) or 3-D (n_chan, n_epochs, n_freqs))

    .. versionadded:: 0.2.0

    Parameters
    ----------
    psd : :py:class:`numpy.ndarray`
        Power spectral density of data, in uV^2/Hz. Must be a N-D array of shape (..., n_freqs).
        See :py:func:`scipy.signal.welch` for more details.
    freqs : :py:class:`numpy.ndarray`
        Array of frequencies. Must be a 1-D array of shape (n_freqs,)
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the lower and upper
        frequencies, as well as the band name (e.g. (0.5, 4, 'Delta')).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    -------
    bandpowers : :py:class:`numpy.ndarray`
        Bandpower array of shape *(n_bands, ...)*.
    """
    # Type checks
    assert isinstance(bands, list), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, "freqs must be a 1-D array of shape (n_freqs,)"
    assert psd.shape[-1] == freqs.shape[-1], "n_freqs must be last axis of psd"

    # Extract frequencies of  只留下有意义的频率
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]

    # Trim PSD to frequencies of interest
    psd = psd[..., idx_good_freq]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        msg = (
            "There are negative values in PSD. This will result in incorrect "
            "bandpower values. We highly recommend working with an "
            "all-positive PSD. For more details, please refer to: "
            "https://github.com/raphaelvallat/yasa/issues/29"
        )
        print(msg)
    # Calculate total power
    total_power = simpson(psd, dx=res, axis=-1)
    total_power = total_power[np.newaxis, ...]

    # Initialize empty array
    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float64)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simpson(psd[..., idx_band], dx=res, axis=-1)
        # simpson: 一种积分

    if relative:
        bp /= total_power
    return bp

@BaseWorker.register("psg.eeg_power_distribution")
class EEGPowerDistributionWorker(BaseWorker):
    def __init__(
        self, path: str, seg_duration: float, seg_overlap_ratio: float, to_band: bool = False
    ) -> None:
        self.path = path
        self.seg_duration = seg_duration
        self.seg_overlap_ratio = seg_overlap_ratio
        self.to_band = to_band

        self.preferences = [
            # 还需要细化吗
            ("C3", "C4"),
            ("F3", "F4", "Fp"),
            ("_O1", "_O2"),
            ("O1", "O2"),
        ]

    def __call__(self, state: dict) -> NDArray:
        def parse(eeg: NDArray, fs: NDArray) -> NDArray[np.float32]:
            # utils.update_duration(state, int(eeg.shape[0] / fs))
            result =  get_eeg_features(
                eeg, int(fs), self.seg_duration, self.seg_overlap_ratio, self.to_band
            )
            utils.update_duration(state, round(((result.shape[1] - 1)* (1 - self.seg_overlap_ratio) + 1) * self.seg_duration))
            return result
        with np.load(utils.match_first(state, self.path, key=create_sort_key(self.preferences))) as data:
            return parse(**data)