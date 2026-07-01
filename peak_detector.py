"""
peak_detector.py
--------------------------------
Respiration Peak Detector
"""

import numpy as np
from scipy.signal import find_peaks

from config import *


class PeakDetector:

    def __init__(self, fps):

        self.fps = fps

    # =====================================

    def get_peaks(self, signal):

        peaks, _ = find_peaks(

            signal,

            prominence=max(

                np.std(signal) * 0.40,

                0.08

            ),

            distance=int(self.fps * 1.2)

        )

        return peaks

    # =====================================

    def bpm_from_peaks(self, signal):

        peaks = self.get_peaks(signal)

        if len(peaks) < 2:

            return 0

        duration = len(signal) / self.fps

        bpm = len(peaks) * 60 / duration

        return bpm

    # =====================================

    def bpm_from_fft(self, processor, signal):

        freq, fft = processor.fft(signal)

        mask = (

            (freq >= LOWCUT)

            &

            (freq <= HIGHCUT)

        )

        if np.sum(mask) == 0:

            return 0

        freq = freq[mask]

        fft = fft[mask]

        idx = np.argmax(fft)

        return freq[idx] * 60

    # =====================================

    def confidence(self, peak_bpm, fft_bpm):

        if peak_bpm == 0 or fft_bpm == 0:

            return 0

        diff = abs(

            peak_bpm - fft_bpm

        )

        conf = max(

            0,

            100 - diff * 8

        )

        return conf

    # =====================================

    def estimate(self, signal, processor):

        if len(signal) < self.fps * 15:

            return None

        peak_bpm = self.bpm_from_peaks(signal)

        fft_bpm = self.bpm_from_fft(

            processor,

            signal

        )

        if peak_bpm == 0:

            bpm = fft_bpm

        elif fft_bpm == 0:

            bpm = peak_bpm

        else:

            bpm = (

                peak_bpm +

                fft_bpm

            ) / 2

        bpm = np.clip(

            bpm,

            MIN_BPM,

            MAX_BPM

        )

        conf = self.confidence(

            peak_bpm,

            fft_bpm

        )

        return {

            "bpm": round(float(bpm), 1),

            "confidence": round(float(conf), 1),

            "peaks": self.get_peaks(signal)

        }