"""
signal_processing.py
=========================================
Signal Processing for Respiration Detection
"""

import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import detrend
from scipy.signal import savgol_filter

from scipy.fft import rfft
from scipy.fft import rfftfreq

from config import *


class SignalProcessor:

    def __init__(self, fps):

        self.fps = fps

    # =====================================================

    def remove_trend(self, signal):

        if len(signal) < 5:

            return signal

        return detrend(signal)

    # =====================================================

    def remove_outlier(self, signal):

        if len(signal) < 10:

            return signal

        median = np.median(signal)

        mad = np.median(

            np.abs(

                signal - median

            )

        )

        if mad < 1e-6:

            return signal

        score = np.abs(

            signal - median

        ) / mad

        cleaned = signal.copy()

        cleaned[score > 3] = median

        return cleaned

    # =====================================================

    def normalize(self, signal):

        if len(signal) == 0:

            return signal

        mean = np.mean(signal)

        std = np.std(signal)

        if std < 1e-8:

            return signal

        return (

            signal - mean

        ) / std

    # =====================================================

    def butter_bandpass(self, signal):

        if len(signal) < self.fps * 2:

            return signal

        nyquist = self.fps * 0.5

        low = LOWCUT / nyquist

        high = HIGHCUT / nyquist

        b, a = butter(

            FILTER_ORDER,

            [low, high],

            btype="band"

        )

        return filtfilt(

            b,

            a,

            signal

        )

    # =====================================================

    def smooth(self, signal):

        if len(signal) < 21:

            return signal

        return savgol_filter(

            signal,

            21,

            3

        )

    # =====================================================

    def moving_average(

            self,

            signal,

            window=5

    ):

        if len(signal) < window:

            return signal

        kernel = np.ones(window)

        kernel /= window

        return np.convolve(

            signal,

            kernel,

            mode="same"

        )

    # =====================================================

    def fft(self, signal):

        if len(signal) < self.fps * 5:

            return None, None

        freq = rfftfreq(

            len(signal),

            1 / self.fps

        )

        amp = np.abs(

            rfft(signal)

        )

        return freq, amp

    # =====================================================

    def dominant_frequency(self, signal):

        freq, amp = self.fft(signal)

        if freq is None:

            return None

        mask = (

            (freq >= LOWCUT)

            &

            (freq <= HIGHCUT)

        )

        if np.sum(mask) == 0:

            return None

        idx = np.argmax(

            amp[mask]

        )

        return freq[mask][idx]

    # =====================================================

    def estimate_bpm_fft(self, signal):

        f = self.dominant_frequency(signal)

        if f is None:

            return None

        return f * 60

    # =====================================================

    def process(self, signal):

        signal = np.asarray(

            signal,

            dtype=np.float32

        )

        signal = self.remove_trend(

            signal

        )

        signal = self.remove_outlier(

            signal

        )

        signal = self.normalize(

            signal

        )

        signal = self.butter_bandpass(

            signal

        )

        signal = self.smooth(

            signal

        )

        signal = self.moving_average(

            signal

        )

        return signal