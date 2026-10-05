"""Reusable preprocessing transforms for MALDI-TOF spectrum objects."""

import numpy as np
from scipy import sparse
from scipy import ndimage
from scipy.linalg import norm
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from helpers import topf
from helpers.SpectrumObject import SpectrumObject

class MinMaxScaler:
    """Pre-processing function for scaling the intensity of a spectrum.

    Parameters
    ----------
    min : int, optional
        minimum value of the scaled spectrum, by default 0
    max : int, optional
        maximum value of the scaled spectrum, by default 1
    """

    def __init__(self, min=0, max=1):
        """Store the target intensity range."""
        self.min = min
        self.max = max

    def __call__(self, SpectrumObj):
        """Return a spectrum with intensity linearly scaled into the target range."""
        s = SpectrumObject(
            intensity=self.min + (SpectrumObj.intensity - SpectrumObj.intensity.min()) * (
                self.max - self.min) / (SpectrumObj.intensity.max() - SpectrumObj.intensity.min()),
            mz=SpectrumObj.mz,
        )
        return s

class StdThresholder:
    """
    Zeros out intensity values below a threshold defined as a multiple of the standard deviation.

    Parameters
    ----------
    factor : float
        Multiplier for the standard deviation. Intensities below (factor * std) are set to zero.
    """
    def __init__(self, factor=2.0):
        """Store the standard-deviation multiplier used as threshold."""
        self.factor = factor

    def __call__(self, SpectrumObj):
        """Zero intensity values below `factor * std(intensity)`."""
        threshold = self.factor * np.std(SpectrumObj.intensity)
        new_intensity = np.where(SpectrumObj.intensity >= threshold, SpectrumObj.intensity, 0.0)
        return SpectrumObject(intensity=new_intensity, mz=SpectrumObj.mz)
    
class LogScaler:
    """Pre-processing function for scaling the intensity of a spectrum using logarithm.

    Parameters
    ----------
    base : int, optional
        base of the logarithm, by default 10
    """

    def __init__(self, base=10):
        """Store the logarithm base."""
        self.base = base

    def __call__(self, SpectrumObj):
        """Return a spectrum with log-scaled intensities."""
        s = SpectrumObject(
            intensity=np.log(SpectrumObj.intensity + 1) / np.log(self.base),
            mz=SpectrumObj.mz,
        )
        return s


class Binner:

    """Pre-processing function for binning spectra in equal-width bins.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    step : int, optional
        width of every bin, by default 3
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, step=3, aggregation="sum"):
        """Create equal-width m/z bins and choose an aggregation statistic."""
        self.bins = np.arange(start, stop + 1e-8, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation
        self.step = step

    def __call__(self, SpectrumObj):
        """Aggregate spectrum intensities into the configured m/z bins."""
        if len(SpectrumObj.mz) == 0:
            return SpectrumObject(intensity=np.zeros_like(self.mz_bins, dtype=float), mz=self.mz_bins)
        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s


class AdaptiveBinner:
    """Pre-processing function for binning spectra with ppm-scaled bin widths.

    The bin edges are deterministic for a configured ``start``, ``stop`` and
    ``ppm_tolerance``, so every processed spectrum shares the same output axis
    while bin widths increase with m/z.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    ppm_tolerance : float, optional
        relative bin width expressed in parts-per-million, by default 2000
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, ppm_tolerance=2000, aggregation="sum"):
        """Create ppm-scaled m/z bins and choose an aggregation statistic."""
        if ppm_tolerance <= 0:
            raise ValueError("ppm_tolerance must be > 0")
        if stop <= start:
            raise ValueError("stop must be > start")

        growth_factor = 1.0 + float(ppm_tolerance) * 1e-6
        edges = [float(start)]
        current = float(start)

        while current < float(stop):
            next_edge = current * growth_factor
            if next_edge <= current:
                raise ValueError("AdaptiveBinner produced a non-increasing bin edge sequence")
            edges.append(min(next_edge, float(stop)))
            current = next_edge

        self.bins = np.asarray(edges, dtype=float)
        self.mz_bins = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.agg = aggregation
        self.ppm_tolerance = ppm_tolerance

    def __call__(self, SpectrumObj):
        """Aggregate spectrum intensities into the configured adaptive m/z bins."""
        if len(SpectrumObj.mz) == 0:
            return SpectrumObject(intensity=np.zeros_like(self.mz_bins, dtype=float), mz=self.mz_bins)
        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s




class Normalizer:
    """Pre-processing function for normalizing the intensity of a spectrum.
    Commonly referred to as total ion current (TIC) calibration.

    Parameters
    ----------
    sum : int, optional
        Make the total intensity of the spectrum equal to this amount, by default 1
    """

    def __init__(self, sum=1):
        """Store the target total ion current."""
        self.sum = sum

    def __call__(self, SpectrumObj):
        """Scale intensities so their sum equals the configured target."""
        intensity = np.asarray(SpectrumObj.intensity, dtype=float)
        total = float(np.sum(intensity))
        if not np.isfinite(total) or total <= 0:
            normalized = np.zeros_like(intensity)
        else:
            normalized = intensity / total * self.sum
        return SpectrumObject(intensity=normalized, mz=SpectrumObj.mz)


class Trimmer:
    """Pre-processing function for trimming ends of a spectrum.
    This can be used to remove inaccurate measurements.

    Parameters
    ----------
    min : int, optional
        remove all measurements with mz's lower than this value, by default 2000
    max : int, optional
        remove all measurements with mz's higher than this value, by default 20000
    """

    def __init__(self, min=2000, max=20000):
        """Store the open m/z interval to retain."""
        self.range = [min, max]

    def __call__(self, SpectrumObj):
        """Return only peaks with m/z inside the configured range."""
        indices = (self.range[0] < SpectrumObj.mz) & (SpectrumObj.mz < self.range[1])

        s = SpectrumObject(
            intensity=SpectrumObj.intensity[indices], mz=SpectrumObj.mz[indices]
        )
        return s


class VarStabilizer:
    """Pre-processing function for manipulating intensities.
    Commonly performed to stabilize their variance.

    Parameters
    ----------
    method : str, optional
        function to apply to intensities.
        can be either "sqrt", "log", "log2" or "log10", by default "sqrt"
    """

    def __init__(self, method="sqrt"):
        """Select the variance-stabilizing transform."""
        self.method = method

    def __call__(self, SpectrumObj):
        """Apply the selected transform to spectrum intensities."""
        intensity = np.asarray(SpectrumObj.intensity, dtype=float)
        # Baseline correction can introduce small negatives; clamp before
        # variance stabilization to avoid NaNs in sqrt/log transforms.
        intensity = np.maximum(intensity, 0.0)

        if self.method == "sqrt":
            transformed = np.sqrt(intensity)
        elif self.method == "log":
            transformed = np.log1p(intensity)
        elif self.method == "log2":
            transformed = np.log2(intensity + 1.0)
        elif self.method == "log10":
            transformed = np.log10(intensity + 1.0)
        else:
            raise ValueError(
                f"Unknown variance-stabilization method '{self.method}'. "
                "Expected one of: sqrt, log, log2, log10."
            )

        s = SpectrumObject(intensity=transformed, mz=SpectrumObj.mz)
        return s


class BaselineCorrecter:
    """Pre-processing function for baseline correction (also referred to as background removal).

    Support SNIP, AsLS, ArPLS, TopHat and ConvexHull.
    Some of the code is based on https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    Parameters
    ----------
    method : str, optional
        Which method to use
        either "SNIP", "AsLS", "ArPLS", "TopHat" or "ConvexHull", by default None
    als_lam : float, optional
        lambda value for ALS and ArPLS, by default 1e8
    als_p : float, optional
        p value for ALS and ArPLS, by default 0.01
    als_max_iter : int, optional
        max iterations for ALS and ArPLS, by default 10
    als_tol : float, optional
        stopping tolerance for ALS and ArPLS, by default 1e-6
    snip_n_iter : int, optional
        iterations of SNIP, by default 10
    tophat_window : int, optional
        structuring-element width for TopHat baseline removal, by default 101
    """

    def __init__(
        self,
        method=None,
        als_lam=1e8,
        als_p=0.01,
        als_max_iter=10,
        als_tol=1e-6,
        snip_n_iter=10,
        tophat_window=101,
    ):
        """Store baseline-correction method and algorithm parameters."""
        self.method = method
        self.lam = als_lam
        self.p = als_p
        self.max_iter = als_max_iter
        self.tol = als_tol
        self.n_iter = snip_n_iter
        self.tophat_window = tophat_window

    def __call__(self, SpectrumObj):
        """Subtract the estimated baseline from spectrum intensities."""
        method = str(self.method).strip()
        method_upper = method.upper()

        if "LS" in method_upper:
            baseline = self.als(
                SpectrumObj.intensity,
                method=method,
                lam=self.lam,
                p=self.p,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        elif method_upper == "SNIP":
            baseline = self.snip(SpectrumObj.intensity, self.n_iter)
        elif method_upper == "TOPHAT":
            baseline = self.tophat(SpectrumObj.intensity, self.tophat_window)
        elif method_upper == "CONVEXHULL":
            baseline = self.convex_hull(SpectrumObj.mz, SpectrumObj.intensity)
        else:
            raise ValueError(
                f"Unknown baseline-correction method '{self.method}'. "
                "Expected one of: SNIP, AsLS, ArPLS, TopHat, ConvexHull."
            )

        s = SpectrumObject(
            intensity=SpectrumObj.intensity - baseline, mz=SpectrumObj.mz
        )
        return s

    def als(self, y, method="ArPLS", lam=1e8, p=0.01, max_iter=10, tol=1e-6):
        """Estimate a baseline with asymmetric least squares or ArPLS."""
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(
            D.transpose()
        )  # Precompute this term since it does not depend on `w`

        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        crit = 1
        count = 0
        while crit > tol:
            z = sparse.linalg.spsolve(W + D, w * y)

            if method == "AsLS":
                w_new = p * (y > z) + (1 - p) * (y < z)
            elif method == "ArPLS":
                d = y - z
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                w_new = 1 / (1 + np.exp(np.minimum(2 * (d - (2 * s - m)) / s, 70)))

            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)
            count += 1
            if count > max_iter:
                break
        return z

    def snip(self, y, n_iter):
        """Estimate a baseline using the SNIP algorithm."""
        y_prepr = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        for i in range(1, n_iter + 1):
            rolled = np.pad(y_prepr, (i, i), mode="edge")
            new = np.minimum(
                y_prepr, (np.roll(rolled, i) + np.roll(rolled, -i))[i:-i] / 2
            )
            y_prepr = new
        return (np.exp(np.exp(y_prepr) - 1) - 1) ** 2 - 1

    def tophat(self, y, window):
        """Estimate a baseline with a morphological top-hat opening."""
        window = max(3, int(window))
        if window % 2 == 0:
            window += 1
        return ndimage.grey_opening(np.asarray(y, dtype=float), size=window)

    def convex_hull(self, x, y):
        """Estimate a baseline by interpolating the lower convex hull."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("ConvexHull baseline expects 1D mz/intensity arrays of equal length.")
        if x.size <= 2:
            return y.copy()

        points = list(zip(x, y))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        hull_x = np.asarray([pt[0] for pt in lower], dtype=float)
        hull_y = np.asarray([pt[1] for pt in lower], dtype=float)
        return np.interp(x, hull_x, hull_y)


class Smoother:
    """Pre-processing function for smoothing. Uses Savitzky-Golay filter.

    Parameters
    ----------
    halfwindow : int, optional
        halfwindow of savgol_filter, by default 10
    polyorder : int, optional
        polyorder of savgol_filter, by default 3
    """

    def __init__(self, halfwindow=10, polyorder=3):
        """Store Savitzky-Golay window and polynomial settings."""
        self.window = halfwindow * 2 + 1
        self.poly = polyorder

    def __call__(self, SpectrumObj):
        """Smooth intensities and clamp negative smoothed values to zero."""
        s = SpectrumObject(
            intensity=np.maximum(
                savgol_filter(SpectrumObj.intensity, self.window, self.poly), 0
            ),
            mz=SpectrumObj.mz,
        )
        return s


class CWTPeakDetector:
    """Detect peaks from a continuous wavelet transform response.

    The CWT coefficient matrix is collapsed back to one response per m/z point,
    then local maxima in that response are retained as peaks. This keeps the
    output compatible with downstream 1D binning steps.

    Parameters
    ----------
    widths : list[float], optional
        Wavelet widths, in data points, used to detect peaks at different scales.
    reduction : str, optional
        How to collapse CWT coefficients across widths: "max", "mean" or "sum".
    min_relative_height : float, optional
        Minimum CWT response as a fraction of the maximum response.
    max_number : int, optional
        Maximum number of peaks to keep, prioritizing higher CWT responses.
    output : str, optional
        Peak intensity source: "original" keeps original intensities, "cwt" uses
        the reduced CWT response.
    """

    def __init__(
        self,
        widths=None,
        reduction="max",
        min_relative_height=0.05,
        max_number=None,
        output="original",
    ):
        """Store CWT peak-detection parameters."""
        if widths is None:
            widths = [1, 2, 4, 8, 16]
        widths = np.asarray(widths, dtype=float)
        if widths.ndim != 1 or widths.size == 0 or np.any(widths <= 0):
            raise ValueError("widths must be a non-empty list of positive values")
        if reduction not in {"max", "mean", "sum"}:
            raise ValueError("reduction must be one of: max, mean, sum")
        if min_relative_height < 0:
            raise ValueError("min_relative_height must be >= 0")
        if output not in {"original", "cwt"}:
            raise ValueError("output must be one of: original, cwt")

        self.widths = widths
        self.reduction = reduction
        self.min_relative_height = float(min_relative_height)
        self.max_number = max_number
        self.output = output

    def __call__(self, SpectrumObj):
        """Return peaks detected from the CWT response."""
        intensity = np.asarray(SpectrumObj.intensity, dtype=float)
        mz = np.asarray(SpectrumObj.mz)

        if intensity.ndim != 1 or mz.ndim != 1 or intensity.size != mz.size:
            raise ValueError("CWTPeakDetector expects equal-length 1D mz/intensity arrays.")
        if intensity.size < 3 or np.all(intensity <= 0):
            return SpectrumObject(
                intensity=np.asarray([], dtype=float),
                mz=np.asarray([], dtype=mz.dtype),
            )

        coeffs = []
        for width in self.widths:
            kernel_size = int(max(7, np.ceil(width * 10)))
            kernel_size = min(kernel_size, intensity.size if intensity.size % 2 else intensity.size - 1)
            if kernel_size < 3:
                continue
            if kernel_size % 2 == 0:
                kernel_size -= 1

            x = np.arange(kernel_size, dtype=float) - (kernel_size - 1) / 2
            wavelet = (1.0 - (x / width) ** 2) * np.exp(-(x ** 2) / (2.0 * width ** 2))
            wavelet_norm = np.sqrt(np.sum(wavelet ** 2))
            if wavelet_norm > 0:
                wavelet = wavelet / wavelet_norm
            coeffs.append(np.abs(np.convolve(intensity, wavelet, mode="same")))

        if not coeffs:
            return SpectrumObject(
                intensity=np.asarray([], dtype=float),
                mz=np.asarray([], dtype=mz.dtype),
            )

        coeffs = np.vstack(coeffs)
        if self.reduction == "max":
            response = np.max(coeffs, axis=0)
        elif self.reduction == "mean":
            response = np.mean(coeffs, axis=0)
        else:
            response = np.sum(coeffs, axis=0)

        threshold = self.min_relative_height * float(np.max(response))
        local_maxima = (
            (response[1:-1] >= response[:-2])
            & (response[1:-1] > response[2:])
            & (response[1:-1] >= threshold)
        )
        peak_indices = np.flatnonzero(local_maxima) + 1

        if self.max_number is not None and peak_indices.size > self.max_number:
            strongest = np.argsort(-response[peak_indices], kind="stable")[: self.max_number]
            peak_indices = np.sort(peak_indices[strongest])

        peak_intensity = response[peak_indices] if self.output == "cwt" else intensity[peak_indices]
        return SpectrumObject(intensity=peak_intensity, mz=mz[peak_indices])


class LocalMaximaPeakDetector:
    """
    Detects peaks a la MaldiQuant

    Parameters
    ----------
    SNR : int, optional
        Signal to noise radio. This function computes a SNR value as the median absolute deviation from the median intensity (MAD).
        Only peaks with intensities a multiple of this SNR are considered. By default 2.
    halfwindowsize: int, optional
        half window size, an intensity can only be a peak if it is the highest value in a window. By default 20, for a total window size of 41.
    """

    def __init__(
        self,
        SNR=2,
        halfwindowsize=20,
    ):
        """Store local-window size and signal-to-noise multiplier."""
        self.hw = halfwindowsize
        self.SNR = SNR

    def __call__(self, SpectrumObj):
        """Return peaks that are local maxima above the estimated noise floor."""
        SNR = (
            np.median(np.abs(SpectrumObj.intensity - np.median(SpectrumObj.intensity)))
            * self.SNR
        )

        local_maxima = np.argmax(
            np.lib.stride_tricks.sliding_window_view(
                SpectrumObj.intensity, (int(self.hw * 2 + 1),)
            ),
            -1,
        ) == int(self.hw)
        s_int_local = SpectrumObj.intensity[self.hw : -self.hw][local_maxima]
        s_mz_local = SpectrumObj.mz[self.hw : -self.hw][local_maxima]
        return SpectrumObject(
            intensity=s_int_local[s_int_local > SNR], mz=s_mz_local[s_int_local > SNR]
        )


class LocalSNRPeakDetector:
    """Detect local maxima whose prominence exceeds a local sideband noise estimate.

    Parameters
    ----------
    SNR : float, optional
        Minimum local signal-to-noise ratio, by default 3.
    halfwindowsize : int, optional
        Number of points on each side used for local maximum detection, by default 10.
    noise_halfwindowsize : int, optional
        Local noise radius in points. The central peak window is excluded from
        the noise estimate. Defaults to five times ``halfwindowsize``.
    """

    def __init__(self, SNR=3.0, halfwindowsize=10, noise_halfwindowsize=None):
        """Store the local SNR threshold and window size."""
        if SNR <= 0:
            raise ValueError("SNR must be > 0")
        if halfwindowsize < 1:
            raise ValueError("halfwindowsize must be >= 1")
        if noise_halfwindowsize is None:
            noise_halfwindowsize = int(halfwindowsize) * 5
        if noise_halfwindowsize <= halfwindowsize:
            raise ValueError("noise_halfwindowsize must be greater than halfwindowsize")
        self.SNR = float(SNR)
        self.hw = int(halfwindowsize)
        self.noise_hw = int(noise_halfwindowsize)

    def __call__(self, SpectrumObj):
        """Return local maxima above the local median by the requested SNR."""
        intensity = np.asarray(SpectrumObj.intensity, dtype=float)
        mz = np.asarray(SpectrumObj.mz)
        window_size = self.hw * 2 + 1

        if intensity.ndim != 1 or mz.ndim != 1 or intensity.size != mz.size:
            raise ValueError("LocalSNRPeakDetector expects equal-length 1D mz/intensity arrays.")
        if intensity.size < window_size:
            return SpectrumObject(
                intensity=np.asarray([], dtype=float),
                mz=np.asarray([], dtype=mz.dtype),
            )

        windows = np.lib.stride_tricks.sliding_window_view(intensity, window_size)
        local_maxima = np.argmax(windows, axis=1) == self.hw
        candidate_offsets = np.flatnonzero(local_maxima)
        keep_offsets = []

        global_baseline = float(np.median(intensity))
        global_deviations = np.abs(intensity - global_baseline)
        global_noise = float(np.median(global_deviations) * 1.4826)
        if global_noise <= np.finfo(float).eps:
            positive_global = np.sort(global_deviations[global_deviations > 0])
            if positive_global.size:
                global_noise = float(
                    np.sqrt(np.mean(positive_global[: max(1, positive_global.size // 2)] ** 2))
                )

        for offset in candidate_offsets:
            center_idx = int(offset + self.hw)
            noise_start = max(0, center_idx - self.noise_hw)
            noise_stop = min(intensity.size, center_idx + self.noise_hw + 1)
            left_sideband = intensity[noise_start : center_idx - self.hw]
            right_sideband = intensity[center_idx + self.hw + 1 : noise_stop]
            noise_samples = np.concatenate([left_sideband, right_sideband])
            if noise_samples.size == 0:
                continue

            baseline = float(np.median(noise_samples))
            deviations = np.abs(noise_samples - baseline)
            noise = float(np.median(deviations) * 1.4826)

            if noise <= np.finfo(float).eps:
                positive_deviations = np.sort(deviations[deviations > 0])
                if positive_deviations.size > 1:
                    lower_half = positive_deviations[: max(1, positive_deviations.size // 2)]
                    noise = float(np.sqrt(np.mean(lower_half ** 2)))
                elif positive_deviations.size == 1:
                    noise = global_noise or float(positive_deviations[0] * 0.01)

            signal = float(intensity[center_idx] - baseline)
            if noise > 0 and signal > 0 and signal >= self.SNR * noise:
                keep_offsets.append(offset)

        keep_offsets = np.asarray(keep_offsets, dtype=int)
        center_slice = slice(self.hw, intensity.size - self.hw)
        return SpectrumObject(
            intensity=intensity[center_slice][keep_offsets],
            mz=mz[center_slice][keep_offsets],
        )


class PeakFilter:
    """Pre-processing function for filtering peaks.

    Filters in two ways: absolute number of peaks and height.

    Parameters
    ----------
    max_number : int, optional
        Maximum number of peaks to keep. Prioritizes peaks to keep by height.
        by default None, for no filtering
    min_intensity : float, optional
        Min intensity of peaks to keep, by default None, for no filtering
    """

    def __init__(self, max_number=None, min_intensity=None):
        """Store peak count and minimum-intensity filters."""
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, SpectrumObj):
        """Filter peaks by intensity and/or keep only the strongest peaks."""
        s = SpectrumObject(intensity=SpectrumObj.intensity, mz=SpectrumObj.mz)

        if self.max_number is not None:
            indices = np.argsort(-s.intensity, kind="stable")
            take = np.sort(indices[: self.max_number])

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        if self.min_intensity is not None:
            take = s.intensity >= self.min_intensity

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        return s


class RandomPeakShifter:
    """Pre-processing function for adding random (gaussian) noise to the mz values of peaks.

    Parameters
    ----------
    std : float, optional
        stdev of the random noise to add, by default 1
    """

    def __init__(self, std=1.0):
        """Store the standard deviation of Gaussian m/z noise."""
        self.std = std

    def __call__(self, SpectrumObj):
        """Return a spectrum with Gaussian noise added to m/z positions."""
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.normal(scale=self.std, size=SpectrumObj.mz.shape),
        )
        return s


class UniformPeakShifter:
    """Pre-processing function for adding uniform noise to the mz values of peaks.

    Parameters
    ----------
    range : float, optional
        let each peak shift by maximum this value, by default 1.5
    """

    def __init__(self, range=1.5):
        """Store the maximum absolute uniform m/z shift."""
        self.range = range

    def __call__(self, SpectrumObj):
        """Return a spectrum with uniform noise added to m/z positions."""
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.uniform(
                low=-self.range, high=self.range, size=SpectrumObj.mz.shape
            ),
        )
        return s


class Binarizer:
    """Pre-processing function for binarizing intensity values of peaks.

    Parameters
    ----------
    threshold : float
        Threshold for the intensities to become 1 or 0.
    """

    def __init__(self, threshold):
        """Store the intensity threshold used for binarization."""
        self.threshold = threshold

    def __call__(self, SpectrumObj):
        """Convert intensities into binary peak-presence values."""
        s = SpectrumObject(
            intensity=(SpectrumObj.intensity > self.threshold).astype(
                SpectrumObj.intensity.dtype
            ),
            mz=SpectrumObj.mz,
        )
        return s

class PersistenceTransformer:
    """Pre-processing function for Peak Detection.
    Uses the Persistance Transformation first outlined in https://doi.org/10.1093/bioinformatics/btaa429
    Underlying code is from https://github.com/BorgwardtLab/Topf

    Parameters
    ----------
    extract_nonzero : bool, optional
        whether to extract detected peaks or to keep zeros in, by default False
    """

    def __init__(self, extract_nonzero=False):
        """Choose whether to keep all persistence values or only detected peaks."""
        self.filter = extract_nonzero

    def __call__(self, SpectrumObj):
        """Apply topological persistence peak detection."""
        a = np.stack([SpectrumObj.mz, SpectrumObj.intensity]).T
        b = topf.PersistenceTransformer().fit_transform(a)

        s = SpectrumObject()
        if self.filter:
            peaks = b[:, 1] != 0
            s = SpectrumObject(intensity=b[peaks, 1], mz=b[peaks, 0])
        else:
            s = SpectrumObject(intensity=b[:, 1], mz=b[:, 0])
        return s



class SequentialPreprocessor:
    """Chain multiple preprocessors so that a pre-processing pipeline can be called with one line.

    Example:
    ```python
    preprocessor = SequentialPreprocessor(
        VarStabilizer(),
        Smoother(),
        BaselineCorrecter(method="SNIP"),
        Normalizer(),
        Binner()
    )
    preprocessed_spectrum = preprocessor(spectrum)
    ```
    """

    def __init__(self, *args):
        """Store preprocessing steps in the order they should be applied."""
        self.preprocessors = args

    def __call__(self, SpectrumObj):
        """Apply each preprocessing step sequentially to a spectrum."""
        for step in self.preprocessors:
            SpectrumObj = step(SpectrumObj)
        return SpectrumObj
