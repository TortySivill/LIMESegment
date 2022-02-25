import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle


import sys
sys.path.append('../')
from Utils.constants import TRAIN_FILES, TEST_FILES
from Utils.perturbations import RBPIndividual, zeroPerturb, noisePerturb, blurPerturb



def perturb(perturbation_strategy, ts, index0, index1):
    if perturbation_strategy == 'RBP':
        return RBPIndividual(ts, index0, index1)
    if perturbation_strategy == 'zero':
        return zeroPerturb(ts, index0, index1)
    if perturbation_strategy == 'noise':
        return noisePerturb(ts, index0, index1)
    if perturbation_strategy == 'blur':
        return blurPerturb(ts, index0, index1)


def ASyntheticPerturb(N):
    samples = []
    for _ in range(0,N):
        mu, sigma = 0, 1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 500)
        time = np.arange(0, 500, 1)
        freq = np.sin(time)
        background_signal = freq + noise
        foreground_signal = np.zeros(500)
        time_component1 = np.arange(0, 100, 1)
        freq_component1 = np.sin(time_component1*0.5)
        time_component2 = np.arange(0, 100, 1)
        freq_component2 = np.sin(time_component2*10)
        foreground_signal[400:500] = freq_component2
        example_signal = background_signal + foreground_signal 
        samples.append(np.asarray(example_signal))
    return np.asarray(samples)

def BSyntheticPerturb(N):
    samples = []
    for _ in range(0,N):
        mu, sigma = 0, 1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 500)
        time = np.arange(0, 500, 1)
        freq = np.sin(time)
        background_signal = freq + noise
        foreground_signal = np.zeros(500)
        time_component1 = np.arange(0, 100, 1)
        freq_component1 = np.sin(time_component1*0.5)
        time_component2 = np.arange(0, 100, 1)
        freq_component2 = np.sin(time_component2*10)
        foreground_signal[400:500] = freq_component1
        example_signal = background_signal + foreground_signal 
        samples.append(np.asarray(example_signal))
    return np.asarray(samples)

def ASyntheticLocality(N):
    samples = []
    for _ in range(0,N):
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1000)
        time = np.arange(0, 1000, 1)
        freq = np.sin(time)
        background_signal = freq + noise

        foreground_signal = np.zeros(1000)

        time_component1 = np.arange(0, 100, 1)
        freq_component1 = np.sin(time_component1*0.5)

        time_component2 = np.arange(0, 100, 1)
        freq_component2 = np.sin(time_component2*20)
        
        
        time_component3 = np.arange(0, 200, 1)
        freq_component3 = np.sin(time_component3*10)

        foreground_signal[0:100] = freq_component1
        foreground_signal[300:400] = freq_component2
        foreground_signal[600:800] = freq_component3

        example_signal = background_signal + foreground_signal 
        samples.append(np.asarray(example_signal))
    return np.asarray(samples)

def BSyntheticLocality(N):
    samples = []
    for _ in range(0,N):
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1000)
        time = np.arange(0, 1000, 1)
        freq = np.sin(time)
        background_signal = freq + noise

        foreground_signal = np.zeros(1000)

        time_component1 = np.arange(0, 100, 1)
        freq_component1 = np.sin(time_component1*0.1)

        time_component2 = np.arange(0, 100, 1)
        freq_component2 = np.sin(time_component2*5)
        
        
        time_component3 = np.arange(0, 200, 1)
        freq_component3 = np.sin(time_component3*10)

        foreground_signal[0:100] = freq_component1
        foreground_signal[300:400] = freq_component2
        foreground_signal[600:800] = freq_component3

        example_signal = background_signal + foreground_signal 
        samples.append(np.asarray(example_signal))
    return np.asarray(samples)

def ASyntheticLocalityComplex(N):
    samples = []
    for _ in range(0,N):
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1000)
        time = np.arange(0, 1000, 1)
        freq = np.sin(time)
        background_signal = freq + noise

        foreground_signal = np.zeros(1000)

        time_component1 = np.arange(0, 100, 1)
        freq_component1 = np.sin(time_component1*0.5)

        time_component2 = np.arange(0, 100, 1)
        freq_component2 = np.sin(time_component2*20)
        
        
        time_component3 = np.arange(0, 200, 1)
        freq_component3 = np.sin(time_component3*10)

        foreground_signal[0:100] = freq_component1
        foreground_signal[300:400] = freq_component2
        foreground_signal[600:800] = freq_component3

        example_signal = background_signal + foreground_signal 
        samples.append(np.asarray(example_signal))
    return np.asarray(samples)

def BSyntheticLocalityComplex(N):
    samples = []
    for _ in range(0,N):
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1000)
        time = np.arange(0, 1000, 1)
        freq = np.sin(time)
        background_signal = freq + noise

        foreground_signal = np.zeros(1000)

        time_component1 = np.arange(0, 100, 1)
        freq_component1 = np.sin(time_component1*0.1)

        time_component2 = np.arange(0, 100, 1)
        freq_component2 = np.sin(time_component2*5)
        
        
        time_component3 = np.arange(0, 200, 1)
        freq_component3 = np.sin(time_component3*12)

        foreground_signal[0:100] = freq_component1
        foreground_signal[300:400] = freq_component2
        foreground_signal[600:800] = freq_component3

        example_signal = background_signal + foreground_signal 
        samples.append(np.asarray(example_signal))
    return np.asarray(samples)

def generateSyntheticPerturbReal(train_size):

    A = ASyntheticPerturb(train_size)
    B = ASyntheticPerturb(train_size)
    rbp_B = [perturb('RBP',example, 400, 500) for example in B]
    zero_B = [perturb('zero',example, 400, 500) for example in B]
    noise_B = [perturb('noise',example, 400, 500) for example in B]
    blur_B = [perturb('blur',example, 400, 500) for example in B]

    x_train_original = np.concatenate((A,B),axis=0)
    x_train_rbp = np.concatenate((A,rbp_B),axis=0)
    x_train_zero = np.concatenate((A,zero_B),axis=0)
    x_train_noise = np.concatenate((A,noise_B),axis=0)
    x_train_blur = np.concatenate((A,blur_B),axis=0)
    y_train = np.asarray(list(np.ones(500)) + list(np.zeros(500)))

    x_train_original = x_train_original.reshape((x_train_original.shape[0], x_train_original.shape[1], 1))
    x_train_rbp = x_train_rbp.reshape((x_train_rbp.shape[0], x_train_rbp.shape[1], 1))
    x_train_zero = x_train_zero.reshape((x_train_zero.shape[0], x_train_zero.shape[1], 1))
    x_train_noise = x_train_noise.reshape((x_train_noise.shape[0], x_train_noise.shape[1], 1))
    x_train_blur = x_train_blur.reshape((x_train_blur.shape[0], x_train_blur.shape[1], 1))

    x_train_original, y_train_original = shuffle(x_train_original, y_train.copy(), random_state=0)
    x_train_rbp, y_train_rbp = shuffle(x_train_rbp, y_train.copy(), random_state=0)
    x_train_zero, y_train_zero = shuffle(x_train_zero, y_train.copy(), random_state=0)
    x_train_noise, y_train_noise = shuffle(x_train_noise, y_train.copy(), random_state=0)
    x_train_blur, y_train_blur = shuffle(x_train_blur, y_train.copy(), random_state=0)

    return [[x_train_original, y_train_original],[x_train_rbp, y_train_rbp],[x_train_zero, y_train_zero],[x_train_noise, y_train_noise],[x_train_blur, y_train_blur]]

def generateSynthetic(test_type, train_size, test_size):
    if test_type == 'perturb':
        A = ASyntheticPerturb(train_size)
        B = BSyntheticPerturb(train_size)
    elif test_type == 'locality':
        A = ASyntheticLocality(train_size)
        B = BSyntheticLocality(train_size)
    elif test_type == 'locality_complex':
        A = ASyntheticLocalityComplex(train_size)
        B = BSyntheticLocalityComplex(train_size)
    
    x_train = np.concatenate((A,B),axis=0)
    y_train = np.asarray(list(np.ones(train_size)) + list(np.zeros(train_size)))
    
    if test_type == 'perturb':
        A = ASyntheticPerturb(test_size)
        B = BSyntheticPerturb(test_size)
    elif test_type == 'locality':
        A = ASyntheticLocality(test_size)
        B = BSyntheticLocality(test_size)
    elif test_type == 'locality_complex':
        A = ASyntheticLocalityComplex(test_size)
        B = BSyntheticLocalityComplex(test_size)

    x_test = np.concatenate((A,B),axis=0)
    y_test = np.asarray(list(np.ones(test_size)) + list(np.zeros(test_size)))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    return x_train, y_train, x_test, y_test

def loadUCRDataID(index, normalize_timeseries=False, verbose=True):

    """
    Loads a Univaraite UCR Dataset indexed by `utils.constants`.

    Args:
        index: Integer index, set inside `utils.constants` that refers to the
            dataset.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        verbose: Whether to describe the dataset being loaded.

    Returns:
        A tuple of shape (X_train, y_train, X_test, y_test, is_timeseries).
        For legacy reasons, is_timeseries is always True.
    """
    assert index < len(TRAIN_FILES), "Index invalid. Could not load dataset at %d" % index
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

    if os.path.exists(TRAIN_FILES[index]):
        df = pd.read_csv(TRAIN_FILES[index], header=None, encoding='latin-1')

    elif os.path.exists(TRAIN_FILES[index][1:]):
        df = pd.read_csv(TRAIN_FILES[index][1:], header=None, encoding='latin-1')

    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))

    is_timeseries = True # assume all input data is univariate time series


    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)


    # fill all missing columns with 0
    df.fillna(0, inplace=True)


    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_train = df[[0]].values
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_train = df.values

    if is_timeseries:
        X_train = X_train[:, :, np.newaxis]

    if os.path.exists(TEST_FILES[index]):
        df = pd.read_csv(TEST_FILES[index], header=None, encoding='latin-1')

    elif os.path.exists(TEST_FILES[index][1:]):
        df = pd.read_csv(TEST_FILES[index][1:], header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (TEST_FILES[index]))

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    df.fillna(0, inplace=True)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_test = df[[0]].values
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = df.values

    if is_timeseries:
        X_test = X_test[:, :, np.newaxis]

    return X_train, y_train, X_test, y_test, nb_classes               