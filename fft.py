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

import scipy.signal as sp
from scipy import optimize, interpolate
import obspy.signal.filter as ob

def freq_to_key(freq):
    return round(12*np.log2(freq/440)+49)

def key_to_freq(key):
    return 2**((key-49)/12)*440



def test_func(x, a, b):
    #return a * np.sin(b * x + c)
    return a * np.sin(b * x)

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
    # seconds = 5  # Note duration of 3 seconds
# chunk_size = 850
chunk_size = 512 #850
chunk_length_secs = chunk_size / fs
window_size = 8000 #4000 

low_key = 20
high_key = 100

scan_key = 1;

myrecording = myrecording.data


def analyze_chunk(chunk_num, data, acc):
    wavdata = data[chunk_num*chunk_size:chunk_num*chunk_size + window_size, 0]
    windowed_data = np.hamming(len(wavdata)) * wavdata

    power = np.abs(np.fft.rfft(windowed_data))
    power_harmonics = squish(power, 5)
    freqs = np.fft.rfftfreq(wavdata.size, d=1/fs)
    
    idx_upper_bound = int(len(wavdata)*key_to_freq(high_key)/fs)
    max_power = np.max(power_harmonics[:idx_upper_bound])
    max_power2 = np.max(wavdata[:])
    best_freq_indx = np.argmax(power_harmonics[:idx_upper_bound])
    best_freq = freqs[best_freq_indx]
    if (int(round(best_freq)) != 0):
        best_key = freq_to_key(best_freq)
    else:
        best_key = 5
    length = 2 * np.pi * key_to_freq(best_key) * chunk_length_secs
    x = np.linspace(acc, acc + length, chunk_size)
    acc += length
    #x_data = np.linspace(chunk_num*chunk_size,(chunk_num+1)*chunk_size,chunk_size)
    #y_data = myrecording[chunk_num*chunk_size:(chunk_num+1)*chunk_size,0]
    #print(y_data)
    #params, params_covariance = optimize.curve_fit(test_func, x_data, y_data,bounds=((0, 0,0),(100000,0.05, 2*np.pi) ) ,maxfev=1000)
    
    #if (int(round(best_freq)) != 0):
        #print(1/best_freq)
        #params, params_covariance = optimize.curve_fit(test_func, x_data, y_data,bounds=((0, 0),(10000,0.1) ),p0=[0,1/best_freq], maxfev = 6000)
        #params, params_covariance = optimize.curve_fit(test_func, x_data, y_data,bounds=((0, 0,0),(100000,0.1, 2*np.pi) ),p0=[60,1/best_freq,np.pi], maxfev = 6000)
        #print(max_power2)
        #print(params[0],params[1])
        #chunk_volume = params[0];
        #chunk_volume
    #else:
        #chunk_volume = 0

    chunk_volume = max_power2
    chunk_data = chunk_volume * np.sin(x)
    #chunk_powers.append(chunk_power)

    return key_to_freq(best_key), best_key, best_freq, chunk_volume, chunk_data, acc, freqs, power, power_harmonics, best_freq_indx


#myenvelope = ob.envelope(myrecording[:,0])
#myenvelope = (sp.hilbert(myrecording[:850,0]))
# myrecording = sd.rec(seconds * fs, samplerate=fs, channels=1)
#sd.wait()  # Wait until recording is finished

num_chunks = len(myrecording) // chunk_size

output_data = np.zeros(num_chunks * chunk_size)

accum = 0
chunk_powers = []
chunk_freqs = []
#num_chunks = 10
for chunk in range(0, num_chunks - (window_size // chunk_size) - 1):
    output_freq, output_key, measured_freq, measured_volume, output_sin, accum, freqs, powers, power_harmonics, best_freq_indx = analyze_chunk(chunk, myrecording, accum)
    output_data[chunk * chunk_size:(chunk + 1) * chunk_size] = output_sin
    print(chunk, output_key, measured_volume, output_freq)
    chunk_freqs.append(output_freq)

print("Sample Rate: " + str(fs) + " Hz")
wavio.write("beautifulmusic.wav", output_data, fs, sampwidth=2)


#must work for 100Hz to 5kHz


#872
start_chunk = 161
#chunks_shown = 8
chunks_shown = 0
chunk = start_chunk


output_freq, output_key, measured_freq, measured_volume, output_sin, accum, freqs, powers, power_harmoncs, best_freq_indx = analyze_chunk(chunk, myrecording, accum)
#wavdata = myrecording[chunk * chunk_size:chunk*chunk_size + window_size, 0]

print("Detected freq = " + str(measured_freq))
print(chunk, output_key, measured_volume, output_freq)


plt.figure(1,figsize=[8,8])
plt.subplot(613)
plt.xscale("log")
plt.plot(freqs[:], powers[:fs//2])
plt.axvline(x=freqs[best_freq_indx], color = 'r')

plt.axvline(x=key_to_freq(output_key-1), color = 'b', linewidth=0.5)
for note in range(low_key, high_key):
    plt.axvline(x=key_to_freq(note), color = 'b', linewidth=0.5)

plt.subplot(612)
plt.plot(output_data[(start_chunk-chunks_shown)*chunk_size:(start_chunk+1+chunks_shown)*chunk_size])
plt.subplot(614)
plt.plot(myrecording[start_chunk*chunk_size:(start_chunk)*chunk_size+window_size])
#plt.plot(wavdata[:])
#plt.plot(myrecording[:850,0])
plt.subplot(615)
plt.plot(myrecording[:,0])
#plt.plot(myenvelope[:850])
#plt.plot(x_data, test_func(x_data, params[0], params[1], params[2]),label='Fitted function')
plt.axvline(x=(start_chunk+0.5)*chunk_size, color = 'B')
plt.subplot(611)
plt.plot(output_data[:])
plt.axvline(x=(start_chunk+0.5)*chunk_size, color = 'B')

print("shape = " + str(myrecording[:,0].shape))
print(str(freqs[1]-freqs[0]))

plt.subplot(616)
#plt.xscale("log")
#plt.plot(freqs[:], powers[:fs//2])
plt.plot(chunk_freqs)
#plt.plot(power_harmonics)
plt.axvline(x=(start_chunk+0.5), color = 'B', linewidth=0.4)

plt.show()
# plt.figure(1)
# plt.subplot(211)
# plt.xscale("log")
# #plt.plot(freq[:], power[:22051])
# plt.plot(freq, power)
# plt.subplot(212)
# plt.plot(chunk_data)
#plt.ylabel('some numbers')
