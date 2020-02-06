#   Copyright (C) 2020, Joshua Wentzel
#
#   This program was written to converts audio files to a piano tone 
#   locked version. This was part of a Oregon State University
#   Junior Design Final Project for 2019-2020.
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.mport scipy.ndimage.filters as fil

import scipy.signal as sp
import matplotlib.pyplot as plt
import numpy as np
import wavio
import sys

from scipy.io.wavfile import write

import matplotlib
matplotlib.use('WebAgg')

# matplotlib inline


def freq_to_key(freq):
    return round(12 * np.log2(freq / 440) + 49)


def key_to_freq(key):
    return 2**((key - 49) / 12) * 440


def resize(array, samples):
    array = array[:-(len(array) % samples)]
    return sum([array[n::samples] for n in range(0, samples)]) / samples


def squish(squish_this, num_squishes):
    output = np.copy(squish_this)
    for squish in range(2, num_squishes):
        output[:len(output) // squish] += resize(squish_this, squish)
    return output


myrecording = wavio.read(sys.argv[1])

fs = myrecording.rate  # 44100 samples per second

if len(sys.argv) >= 3:
    start_chunk = int(sys.argv[2])
else:
    start_chunk = 50

CHUNK_SIZE = 256
chunk_length_secs = CHUNK_SIZE / fs
WINDOW_SIZE = 20024  # 4000

LOW_KEY = 20
HIGH_KEY = 100

scan_key = 1

myrecording = [item[0] for item in myrecording.data]
myrecording = np.array(myrecording)


idx_upper_bound = int(WINDOW_SIZE * key_to_freq(HIGH_KEY) / fs)
idx_lower_bound = int(WINDOW_SIZE * key_to_freq(LOW_KEY) / fs)


def analyze_chunk(chunk_num, data, acc):
    wavdata = data[chunk_num * CHUNK_SIZE:chunk_num * CHUNK_SIZE + WINDOW_SIZE]
    windowed_data = sp.windows.general_gaussian(
        len(wavdata), p=1, sig=WINDOW_SIZE) * wavdata

    power = np.abs(np.fft.rfft(windowed_data))
    freq_peaks = []
    power_harmonics = squish(power, 5)
    freqs = np.fft.rfftfreq(wavdata.size, d=1 / fs)

    power_harmonics[:idx_lower_bound] *= 0
    power_harmonics[idx_upper_bound:] *= 0

    freq_peaks = sp.find_peaks(
        power_harmonics[idx_lower_bound:idx_upper_bound], distance=3, height=500000)
    bin_width = abs(freqs[1] - freqs[0])

    new_peak_frequencies = []
    new_peak_magnitudes = []
    new_bins = []
    for peak in freq_peaks[0]:
        a = power_harmonics[peak + idx_lower_bound - 1]
        b = power_harmonics[peak + idx_lower_bound]
        c = power_harmonics[peak + idx_lower_bound + 1]
        bin_offset = 0.5 * (a - b) / (a - 2 * b + c)
        new_peak = b - 0.25 * (a - c) * bin_offset
        new_freq = freqs[peak + idx_lower_bound] + bin_width * bin_offset
        new_peak_frequencies.append(new_freq)
        new_peak_magnitudes.append(new_peak)
        new_bins.append(bin_offset + peak)

    best_freq = new_peak_frequencies[np.argmax(new_peak_magnitudes)]
    old_best_freq_idx = np.argmax(power_harmonics[:idx_upper_bound])
    old_best_key = freq_to_key(freqs[old_best_freq_idx])

    half = CHUNK_SIZE // 2
    middle = len(wavdata) // 2
    max_power2 = np.max(wavdata[middle - half:middle + half])
    best_freq_indx = np.argmax(power_harmonics[:idx_upper_bound])
    best_key = freq_to_key(best_freq)

    if not (best_key == old_best_key):
        if (old_best_key > best_key):
            print(old_best_key, best_key, chunk_num, "Moved Down!")
        else:
            print(old_best_key, best_key, chunk_num, "Moved Up!")
            # don't trust the new results when the peak goes up
            best_key = old_best_key

    length = 2 * np.pi * key_to_freq(best_key) * chunk_length_secs
    x = np.linspace(acc, acc + length, CHUNK_SIZE)
    acc += length
    chunk_volume = max_power2
    chunk_data = chunk_volume * np.sin(x)

    return key_to_freq(
        best_key), best_key, best_freq, chunk_volume, chunk_data, acc, freqs, power, power_harmonics, best_freq_indx, freq_peaks


def compute_waves(chunk_num, f, p_l, p_c, p_n, acc):
    shared_last = (p_c / 2 + p_l / 2)
    shared_next = (p_n / 2 + p_c / 2)

    p = np.linspace(shared_last, shared_next, CHUNK_SIZE)
    length = 2 * np.pi * f * chunk_length_secs
    x = np.linspace(acc, acc + length, CHUNK_SIZE)
    acc += length
    return np.sin(x) * p, acc


num_chunks = len(myrecording) // CHUNK_SIZE

output_data = np.zeros(num_chunks * CHUNK_SIZE)

accum = 0
chunk_powers = []
chunk_freqs = []
pwr_harmonics = []
limit = num_chunks - (WINDOW_SIZE // CHUNK_SIZE) - 1
for chunk in range(0, limit):
    output_freq, output_key, measured_freq, measured_volume, output_sin, accum, freqs, powers, power_harmonics, best_freq_indx, peaks = analyze_chunk(
        chunk, myrecording, accum)
    pwr_harmonics.append(power_harmonics)
    chunk_freqs.append(output_freq)
    chunk_powers.append(abs(measured_volume))

old_chunk_freqs = np.copy(chunk_freqs)
chunk_freqs = sp.medfilt(chunk_freqs, (fs // WINDOW_SIZE) * 30 + 1)

accum = 0
for chunk in range(0, limit):
    frequency = chunk_freqs[chunk]
    next_power = 0
    last_power = 0

    current_power = chunk_powers[chunk]
    if (chunk + 1 < limit):
        next_power = chunk_powers[chunk + 1]
    if (chunk - 1 >= 0):
        last_power = chunk_powers[chunk - 1]
    sin_data, accum = compute_waves(
        chunk, frequency, last_power, current_power, next_power, accum)

    output_data[chunk * CHUNK_SIZE:(chunk + 1) * CHUNK_SIZE] = sin_data


print("Sample Rate: " + str(fs) + " Hz")
wavio.write("output.wav", output_data, fs, sampwidth=2)

chunks_shown = 0
chunk = start_chunk

output_freq, output_key, measured_freq, measured_volume, output_sin, accum, freqs, powers, power_harmonics, best_freq_indx, peaks = analyze_chunk(
    chunk, myrecording, accum)

print("Detected freq = " + str(measured_freq))
print(chunk, output_key, measured_volume, output_freq)


fig = plt.figure(1, figsize=[8, 8.9])
plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0.05)
plt.subplot(717)
plt.plot(output_data[:])
plt.axvline(x=(start_chunk + 0.5) * CHUNK_SIZE, color='b')

print("shape = " + str(myrecording.shape))
print("Space between frequency bins: " + str(freqs[1] - freqs[0]) + " Hz")

plt.subplot(716)
plt.yscale("log")
for note in range(16, 108, 12):
    plt.hlines(
        y=key_to_freq(note),
        color='black',
        linewidth=0.5,
        xmin=0,
        xmax=num_chunks)
plt.hlines(
    y=[key_to_freq(LOW_KEY),key_to_freq(HIGH_KEY)],
    color='r',
    linewidth=0.5,
    xmin=0,
    xmax=num_chunks)
plt.axvline(x=(start_chunk + 0.5), color='b', linewidth=0.4)
plt.plot(chunk_freqs)

plt.subplot(715)
plt.yscale("log")
for note in range(16, 108, 12):
    plt.hlines(
        y=key_to_freq(note),
        color='black',
        linewidth=0.5,
        xmin=0,
        xmax=num_chunks)
plt.hlines(
    y=[key_to_freq(LOW_KEY),key_to_freq(HIGH_KEY)],
    color='r',
    linewidth=0.5,
    xmin=0,
    xmax=num_chunks)
plt.axvline(x=(start_chunk + 0.5), color='b', linewidth=0.4)
plt.plot(old_chunk_freqs)

plt.subplot(713)
plt.xscale("log")
plt.plot(freqs[:], powers[:fs // 2])
plt.axvline(x=freqs[best_freq_indx], color='r')

for note in range(16, 108, 12):
    plt.axvline(x=key_to_freq(note), color='black', linewidth=0.5)

plt.axvline(x=key_to_freq(LOW_KEY), color='r', linewidth=0.5)
plt.axvline(x=key_to_freq(HIGH_KEY), color='r', linewidth=0.5)


plt.subplot(714)
plt.xscale("log")
plt.xlim([key_to_freq(LOW_KEY), key_to_freq(HIGH_KEY)])
plt.plot(freqs[:], power_harmonics)
for note in range(16, 108, 12):
    plt.axvline(x=key_to_freq(note), color='black', linewidth=0.5)
for peak in peaks[0]:
    plt.axvline(x=freqs[peak + idx_lower_bound],
                color='green', linewidth=1, linestyle='--')
print(peaks[0])

plt.axvline(x=key_to_freq(LOW_KEY), color='r', linewidth=0.5)
plt.axvline(x=key_to_freq(HIGH_KEY), color='r', linewidth=0.5)
plt.axvline(x=freqs[best_freq_indx], color='r')

plt.subplot(712)
plt.plot(myrecording[start_chunk *
                     CHUNK_SIZE:(start_chunk) *
                     CHUNK_SIZE +
                     WINDOW_SIZE])
plt.subplot(711)
plt.plot(myrecording)
plt.axvline(x=(start_chunk + 0.5) * CHUNK_SIZE, color='Blue')
plt.show()
