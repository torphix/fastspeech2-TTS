import torch
import numpy as np
import librosa
import librosa.util as librosa_util
from scipy.signal import get_window
from scipy.interpolate import interp1d
from scipy.io import wavfile
import pyworld as pw


def build_mel_basis(sample_rate, n_fft, num_mels, fmin):
    return librosa.filters.mel(
        sample_rate,
        n_fft,
        n_mels=num_mels,
        fmin=fmin,
    )


def linear_to_mel(spectrogram, sample_rate, n_fft, num_mels, fmin):
    mel_basis = build_mel_basis(sample_rate, n_fft, num_mels, fmin)
    return np.dot(mel_basis, spectrogram)


def normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def melspectrogram(y, sample_rate, n_fft, n_mels, 
                   f_min, hop_length, win_length, min_level_db):
    D = stft(y, n_fft, hop_length, win_length)
    S = amp_to_db(linear_to_mel(np.abs(D), sample_rate, n_fft, n_mels, f_min))
    return normalize(S, min_level_db)


def stft(y, n_fft, hop_length, win_length):
    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    
def energy(y, n_fft, hop_length, win_length):
    # Extract energy
    S = librosa.magphase(stft(y, n_fft, hop_length, win_length))[0]
    e = np.sqrt(np.sum(S ** 2, axis=0))  # np.linalg.norm(S, axis=0)
    return e.squeeze()  # (Number of frames) => (654,)

def energy_phoneme_averaging(energy, duration):
        # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            energy[i] = np.mean(energy[pos : pos + d])
        else:
            energy[i] = 0
        pos += d
    energy = energy[: len(duration)]
    return energy

def pitch(wav, sample_rate, hop_length, duration):
    # Extract Pitch/f0 from raw waveform using PyWORLD
    wav = wav.astype(np.float64)
    """
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    """
    f0, timeaxis = pw.dio(
        wav,
        sample_rate,
        frame_period=hop_length / sample_rate * 1000,
    )  # For hop size 256 frame period is 11.6 ms
    pitch = pw.stonemask(wav.astype(np.float64), f0, timeaxis, sample_rate)
    pitch = pitch[: len(duration)]    
    return f0  # (Number of Frames) = (654,)


def pitch_phoneme_averaging(pitch, duration):
    # perform linear interpolation
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            pitch[i] = np.mean(pitch[pos : pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[: len(duration)]
    return pitch


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C



def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x

