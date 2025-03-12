# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Derived from torchaudio (https://github.com/pytorch/audio/)
"""

import math
import torch


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


_SAMPLE_RATE = 16000
_STFT_WINDOW_LENGTH_SECONDS = 0.025
_STFT_HOP_LENGTH_SECONDS = 0.010
_MEL_MIN_HZ = 125
_MEL_MAX_HZ = 7500
_NUM_BANDS = 64
_LOG_OFFSET = 0.01
_EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
_EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.


def _frame(data, window_length, hop_length):
    num_samples = data.shape[0]
    num_frames = 1 + int(math.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.stride()[0] * hop_length,) + data.stride()
    return torch.as_strided(data, shape, strides)


def _stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = _frame(signal, window_length, hop_length)
    window = torch.hann_window(window_length, periodic=True).to(signal.device)
    windowed_frames = frames * window
    return torch.abs(torch.fft.rfft(windowed_frames, int(fft_length)))


def _hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * torch.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _spectrogram_to_mel_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    audio_sample_rate=8000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
):
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" % (lower_edge_hertz, upper_edge_hertz))

    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" % (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = torch.linspace(0.0, nyquist_hertz, num_spectrogram_bins)

    spectrogram_bins_mel = _hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = torch.linspace(
        _hertz_to_mel(torch.tensor(lower_edge_hertz)),
        _hertz_to_mel(torch.tensor(upper_edge_hertz)),
        num_mel_bins + 2,
    )
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = torch.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)

        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = torch.maximum(torch.tensor(0.0), torch.minimum(lower_slope, upper_slope))

    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def _log_mel_spectrogram(
    data,
    audio_sample_rate=8000,
    log_offset=0.0,
    window_length_secs=0.025,
    hop_length_secs=0.010,
    **kwargs,
):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(math.ceil(math.log(window_length_samples) / math.log(2.0)))

    spectrogram = _stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples,
    )
    mel_spectrogram = torch.matmul(
        spectrogram,
        _spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate,
            **kwargs,
        ).to(spectrogram),
    )
    return torch.log(mel_spectrogram + log_offset)


def _waveform_to_examples(data):
    # Compute log mel spectrogram features, with shape (n_frame, n_mel)
    log_mel = _log_mel_spectrogram(
        data,
        audio_sample_rate=_SAMPLE_RATE,
        log_offset=_LOG_OFFSET,
        window_length_secs=_STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=_STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=_NUM_BANDS,
        lower_edge_hertz=_MEL_MIN_HZ,
        upper_edge_hertz=_MEL_MAX_HZ,
    )

    # Frame features into examples, with shape (n_example, n_frame, n_mel)
    features_sample_rate = 1.0 / _STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(_EXAMPLE_WINDOW_SECONDS * features_sample_rate))

    example_hop_length = int(round(_EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = _frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)

    # (n_example, 1, n_frame, n_mel)
    return log_mel_examples.unsqueeze(1)


class VGGishInputProcessor:
    """Converts raw waveforms to batches of examples to use as inputs to VGGish."""

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): waveform, with shape `(T,)`.
            sample_rate (int): sample rate of waveform in hertz.

        Returns:
            torch.Tensor: batch of examples to pass to VGGish, with shape `(n_example, 1, n_frame, 64)`.
        """
        if len(input.shape) != 1:
            raise ValueError("input waveform must have dimension of 1.")
        return _waveform_to_examples(input)
