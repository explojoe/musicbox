import numpy as np
import simpleaudio as sa
import wavio
import sys

import sounddevice as sd
from scipy.io.wavfile import write

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
#matplotlib inline

def freq_to_key(freq):
    return round(12*np.log2(freq/440)+49)

def key_to_freq(key):
    return 2**((key-49)/12)*440

myrecording = wavio.read(sys.argv[1])

fs = myrecording.rate  # 44100 samples per second
    # seconds = 5  # Note duration of 3 seconds
# chunk_size = 850
chunk_size = 512
window_size = 3000

low_key = 20
high_key = 100

myrecording = myrecording.data

# myrecording = sd.rec(seconds * fs, samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

num_chunks = len(myrecording) // chunk_size

output_data = np.zeros(num_chunks * chunk_size)

accum = 0
for chunk in range(0, num_chunks):
    # print("Sample width: " + str(wav.sampwidth))
    # print("Sample Rate: " + str(wav.rate))
    wavdata = myrecording[chunk * chunk_size:chunk*chunk_size + window_size, 0]

    windowed_data = np.hamming(len(wavdata)) * wavdata
    # windowed_data = wavdata
    power = np.abs(np.fft.rfft(windowed_data))

    #plt.plot(freq[:22051], power[:22051])
    # plt.plot(output_data[len(output_data)//2:len(output_data)//2 + chunk_size * 4])
    # plt.show()
    # plt.figure(1)
    # plt.subplot(211)
    # plt.xscale("log")
    # #plt.plot(freq[:], power[:22051])
    # plt.plot(freq, power)
    # plt.subplot(212)
    # plt.plot(chunk_data)
    #plt.ylabel('some numbers')

    # freqs = np.fft.rfftfreq(wavdata.size, d=1/fs)

    note_weights = []
    for note in range(low_key, high_key):
        freq_fundamental = key_to_freq(note)
        weight = 0
        for harmonic in range(1, 3):
            freq = harmonic * freq_fundamental
            index = int((window_size / fs) * freq)
            # index = freqs.where(freq)
            if index < len(power):
                weight += power[index]
        note_weights.append(weight)
    best_note = np.argmax(note_weights) + low_key
    best_note_weight = note_weights[best_note - low_key]
    best_freq = key_to_freq(best_note)
    best_freq_power = power[int((window_size / fs) * best_freq)]
    
    chunk_length_secs = chunk_size / fs
    length = 2 * np.pi * best_freq * chunk_length_secs
    x = np.linspace(accum, accum + length, chunk_size)
    accum += length
    # x = np.linspace(chunk * chunk_length_secs, (chunk + 1) * chunk_length_secs, chunk_size)
    chunk_data = (2 ** 15) * np.sin(x)
    #chunk_data = best_freq_power * np.sin(x)
    print(chunk, best_note, best_note_weight, best_freq)

    if best_note_weight > 000:
        output_data[chunk * chunk_size:(chunk + 1) * chunk_size] = chunk_data


print("Sample Rate: " + str(fs) + " Hz")
wavio.write("beautifulmusic.wav", output_data, fs, sampwidth=2)


#must work for 100Hz to 5kHz

start_chunk = 1126
#chunks_shown = 8
chunks_shown = 1
wavdata = myrecording[chunk * chunk_size:chunk*chunk_size + window_size, 0]

windowed_data = np.hamming(len(wavdata)) * wavdata
    # windowed_data = wavdata
power = np.abs(np.fft.rfft(windowed_data))
freqs = np.fft.rfftfreq(wavdata.size, d=1/fs)

note_weights = []
for note in range(low_key, high_key):
    freq_fundamental = key_to_freq(note)
    weight = 0
    for harmonic in range(1, 2):
        freq = harmonic * freq_fundamental
        index = int((window_size / fs) * freq)
        # index = freqs.where(freq)
        if index < len(power):
            weight += power[index]/(harmonic**7)
            #print(note, harmonic, power[index]/(harmonic**7))
    note_weights.append(weight)
best_note = np.argmax(note_weights) + low_key
best_note_weight = note_weights[best_note - low_key]
best_freq = key_to_freq(best_note)
best_freq_power = power[int((window_size / fs) * best_freq)]

chunk_length_secs = chunk_size / fs
length = 2 * np.pi * best_freq * chunk_length_secs
x = np.linspace(accum, accum + length, chunk_size)
accum += length
# x = np.linspace(chunk * chunk_length_secs, (chunk + 1) * chunk_length_secs, chunk_size)
chunk_data = (2 ** 15) * np.sin(x)
#chunk_data = best_freq_power * np.sin(x)
print(chunk, best_note, best_note_weight, best_freq)

if best_note_weight > 000:
    output_data[chunk * chunk_size:(chunk + 1) * chunk_size] = chunk_data
#plt.plot(freq[:22051], power[:22051])
#plt.plot(output_data[len(output_data)//2:len(output_data)//2 + chunk_size * 8])

plt.figure(1,figsize=[8,8])
plt.subplot(411)
plt.xscale("log")
plt.plot(freqs[:], power[:fs//2])
plt.axvline(x=best_freq, color = 'r')
#plt.axvline(x=key_to_freq(best_note-1), color = 'b', linewidth=0.5)
for note in range(low_key, high_key):
    plt.axvline(x=key_to_freq(note), color = 'b', linewidth=0.5)
#plt.axvline(x=key_to_freq(23), color = 'g', linewidth=2)
#plt.axvline(x=key_to_freq(80), color = 'g', linewidth=2)
plt.subplot(412)

plt.plot(output_data[(start_chunk-chunks_shown)*chunk_size:(start_chunk+chunks_shown)*chunk_size])
plt.subplot(413)
#plt.plot(wavdata[start_chunk*chunk_size:(start_chunk+chunks_shown)*chunk_size])
plt.plot(wavdata[:])
plt.subplot(414)
plt.plot(myrecording[:,0])
plt.show()
# plt.figure(1)
# plt.subplot(211)
# plt.xscale("log")
# #plt.plot(freq[:], power[:22051])
# plt.plot(freq, power)
# plt.subplot(212)
# plt.plot(chunk_data)
#plt.ylabel('some numbers')
