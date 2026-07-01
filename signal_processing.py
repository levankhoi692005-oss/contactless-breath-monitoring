"""
signal_processing.py
--------------------------------
Respiration Signal Processing
"""

import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import detrend


class SignalProcessor:

    def __init__(self, fps):

        self.fps = fps

    # =====================================

    def butter_bandpass(self, signal):

        nyquist = 0.5 * self.fps

        low = LOWCUT / nyquist
        high = HIGHCUT / nyquist

        b, a = butter(

            FILTER_ORDER,

            [low, high],

            btype="band"

        )

        return filtfilt(b, a, signal)

    # =====================================

    def normalize(self, signal):

        signal = signal - np.mean(signal)

        std = np.std(signal)

        if std < 1e-6:
            return signal

        return signal / std

    # =====================================

    def moving_average(self, signal, k=5):

        kernel = np.ones(k) / k

        return np.convolve(

            signal,

            kernel,

            mode="same"

        )

    # =====================================

    def remove_outlier(self, signal):

        median = np.median(signal)

        mad = np.median(

            np.abs(signal - median)

        )

        if mad < 1e-6:
            return signal

        z = np.abs(

            signal - median

        ) / (1.4826 * mad)

        signal[z > 3] = median

        return signal

    # =====================================

    def process(self, signal):

        signal = np.asarray(

            signal,

            dtype=np.float32

        )

        signal = detrend(signal)

        signal = self.remove_outlier(signal)

        signal = self.normalize(signal)

        signal = self.butter_bandpass(signal)

        signal = self.moving_average(signal, 5)

        return signal

    # =====================================

    def fft(self, signal):

        n = len(signal)

        fft = np.abs(

            np.fft.rfft(signal)

        )

        freq = np.fft.rfftfreq(

            n,

            1 / self.fps

        )

        return freq, fft