"""
peak_detector.py
=========================================
Peak Detection for Respiration
"""

import numpy as np

from scipy.signal import find_peaks

from config import *


class PeakDetector:

    def __init__(self, fps):

        self.fps = fps

    # ==================================================

    def detect_peaks(self, signal):

        if len(signal) < self.fps * MIN_SIGNAL_SECONDS:

            return None

        prominence = max(
            np.std(signal) * 0.4,
            0.08
        )

        distance = int(
            self.fps * 1.2
        )

        peaks, properties = find_peaks(

            signal,

            prominence=prominence,

            distance=distance

        )

        return peaks

    # ==================================================

    def bpm_from_peaks(self, signal):

        peaks = self.detect_peaks(signal)

        if peaks is None:

            return None

        if len(peaks) < 2:

            return None

        duration = len(signal) / self.fps

        bpm = (

            len(peaks) * 60

        ) / duration

        return bpm

    # ==================================================

    def bpm_from_fft(

            self,

            processor,

            signal

    ):

        return processor.estimate_bpm_fft(

            signal

        )

    # ==================================================

    def confidence(

            self,

            peak_bpm,

            fft_bpm

    ):

        if peak_bpm is None:

            return 0

        if fft_bpm is None:

            return 0

        diff = abs(

            peak_bpm -

            fft_bpm

        )

        score = max(

            0,

            100 -

            diff * 5

        )

        return score

    # ==================================================

    def validate(self, bpm):

        if bpm is None:

            return None

        if bpm < MIN_BPM:

            return None

        if bpm > MAX_BPM:

            return None

        return bpm

    # ==================================================

    def estimate(

            self,

            signal,

            processor=None

    ):

        peak_bpm = self.bpm_from_peaks(

            signal

        )

        peak_bpm = self.validate(

            peak_bpm

        )

        fft_bpm = None

        if processor is not None:

            fft_bpm = self.bpm_from_fft(

                processor,

                signal

            )

            fft_bpm = self.validate(

                fft_bpm

            )

        if peak_bpm is None:

            if fft_bpm is None:

                return None

            bpm = fft_bpm

            score = 70

        elif fft_bpm is None:

            bpm = peak_bpm

            score = 70

        else:

            bpm = (

                peak_bpm +

                fft_bpm

            ) / 2

            score = self.confidence(

                peak_bpm,

                fft_bpm

            )

        return {

            "bpm": round(

                float(bpm),

                1

            ),

            "peak_bpm": round(

                float(

                    peak_bpm

                    if peak_bpm

                    else 0

                ),

                1

            ),

            "fft_bpm": round(

                float(

                    fft_bpm

                    if fft_bpm

                    else 0

                ),

                1

            ),

            "confidence": round(

                float(score),

                1

            )

        }