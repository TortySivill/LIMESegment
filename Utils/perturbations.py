
from scipy import signal
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def backgroundIdentification(original_signal):
    f, t, Zxx = signal.stft(original_signal,1,nperseg=40)
    frequency_composition_abs = np.abs(Zxx)
    measures = []
    for freq,freq_composition in zip(f,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    weights = 1-(measures/sum(measures))
    dummymatrix = np.zeros((len(f),len(t)))
    dummymatrix[selected_frequency,:] = 1  
    #Option to admit information from other frequency bands
    """dummymatrix = np.ones((len(f),len(t)))
    for i in range(0,len(weights)):
        dummymatrix[i,:] = dummymatrix[i,:] * weights[i]"""
    
    background_frequency = Zxx * dummymatrix
    _, xrec = signal.istft(background_frequency, 1)
    return xrec

def RBP(generated_samples_interpretable, original_signal, segment_indexes):
    generated_samples_raw = []
    xrec = backgroundIdentification(original_signal)
    for sample_interpretable in generated_samples_interpretable:
        raw_signal = original_signal.copy()
        for index in range(0,len(sample_interpretable)-1):
            if sample_interpretable[index] == 0:
                index0 = segment_indexes[index]
                index1 = segment_indexes[index+1]
                raw_signal[index0:index1] = xrec[index0:index1]
        generated_samples_raw.append(np.asarray(raw_signal))
    return np.asarray(generated_samples_raw)

def RBPIndividual(original_signal, index0, index1):
    xrec = backgroundIdentification(original_signal)
    raw_signal = original_signal.copy()
    raw_signal[index0:index1] = xrec[index0:index1]
    return raw_signal

def zeroPerturb(original_signal, index0, index1):
    new_signal = original_signal.copy()
    new_signal[index0:index1] = np.zeros(100)
    return new_signal

def noisePerturb(original_signal, index0, index1):
    new_signal = original_signal.copy()
    new_signal[index0:index1] = np.random.randint(-100,100,100).reshape(100)
    return new_signal 


def blurPerturb(original_signal, index0, index1):
    new_signal = original_signal.copy()
    new_signal[index0:index1] = gaussian_filter(new_signal[index0:index1],np.std(original_signal))
    return new_signal

