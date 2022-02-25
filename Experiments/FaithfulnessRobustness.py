import numpy as np 
import matplotlib.pyplot as plt
import stumpy
import pandas as pd
from scipy import signal
from sklearn.linear_model import Ridge
from scipy.spatial import distance

import sys
sys.path.append('../')
from Utils.explanations import LIMESegment, NEVES, LEFTIST, NNSegment, RBP, background_perturb
from Utils.data import loadUCRDataID
from Utils.models import *
from Utils.metrics import *

"""
Script prints the Faithfulness and Robustness Scores for each explanation method, UCR dataset 
classification model as recorded in the paper "LIMESegment."

Load each dataset from Data folder storing each train/test set in dictionary: datasets
Train each classifier on each dataset, storing trained model in dictionary: trined_models
Generate explantions for each dataset and model and save resulting explanations in dictionary: explanations
Evaluate Robustness and Faithfulness for each dataset, model and explanation set and save in dictionary: evaluation_metrics

"""

def reshaper(x,j):
    if j == 0:
        return x.reshape(x.shape[0])
    else:
        return x


dataset_map = [('Coffee', 0),
            	('Strawberry', 1),
                   ('GunPointOldVersusYoung', 2),
                   ('HandOutlines', 3),
                    ('yoga', 4),
                    ('ECG200', 5),
                    ('GunPointMaleVersusFemale', 6),
                    ('DodgerLoopGame', 7),
                    ('Chinatown', 8),
                    ('FreezerSmallTrain', 9),
                    ('HouseTwenty', 10),
                    ('WormsTwoClass', 11)
                    ]

datasets = {}
for data_idx in dataset_map:
    datasets[data_idx[0]] = loadUCRDataID(data_idx[1])

"""Datasets['Coffee'] : Train x, Train y, Test x, Test y

Train x : N x T x 1
Train y: N x 1 
Test x: N' x T x 1
Test y: N' x 2"""

BATCH_SIZES = [8, 64, 32, 64, 64, 8, 16, 4, 4, 4, 8, 16]
WINDOW_SIZES = [20, 20, 10, 100, 50, 10, 10, 20, 3, 10, 200, 100]
CPS = [5, 5, 4, 8, 5, 2, 4, 5, 2, 5, 8, 6]
MODEL_TYPES = ['class','proba','proba']

models = ['knn','cnn','lstmfcn']
trained_models = {}
i = 0
for data_idx in datasets.keys():
    print(data_idx)
    trained_models[data_idx] = {}
    trained_models[data_idx]['knn'] = train_KNN_model(datasets[data_idx][0],datasets[data_idx][1])
    model_cnn = make_CNN_model(datasets[data_idx][0].shape[1:])
    trained_models[data_idx]['cnn'] = train_CNN_model(model_cnn,
                                                      datasets[data_idx][0],
                                                      datasets[data_idx][1],
                                                      epochs=100,
                                                      batch_size=BATCH_SIZES[i])[0]
    model_lstmfcn = make_LSTMFCN_model(datasets[data_idx][0].shape[1])
    trained_models[data_idx]['lstmfcn'] = train_LSTMFCN_model(model_lstmfcn,
                                                      datasets[data_idx][0],
                                                      datasets[data_idx][1],
                                                      datasets[data_idx][2],
                                                      datasets[data_idx][3],
                                                      epochs=100,
                                                      batch_size=BATCH_SIZES[i])
    i = i + 1

explanations = {}
i = 0 
noisy_explanations = {} # For Robustness later
for data_idx in datasets.keys():
    print('processing explanations for: ' + str(data_idx) + '\n')
    explanations[data_idx] = {}
    noisy_explanations[data_idx] = {}
    j = 0
    for model_idx in trained_models[data_idx].keys():
        explanations[data_idx][model_idx] = {}
        noisy_explanations[data_idx][model_idx] = {}
        explanation_set = datasets[data_idx][2][0:10]
        explanations[data_idx][model_idx]['LS'] = [LIMESegment(reshaper(x,j), trained_models[data_idx][model_idx], model_type=MODEL_TYPES[j], window_size=WINDOW_SIZES[i], cp=CPS[i]) for x in explanation_set]
        explanations[data_idx][model_idx]['N'] = [NEVES(reshaper(x,j), trained_models[data_idx][model_idx], datasets[data_idx][0], model_type=MODEL_TYPES[j]) for x in explanation_set]
        explanations[data_idx][model_idx]['LF'] = [LEFTIST(reshaper(x,j), trained_models[data_idx][model_idx], datasets[data_idx][0], model_type=MODEL_TYPES[j],) for x in explanation_set]
        
        noisy_set = np.asarray([add_noise(x) for x in explanation_set])
        
        noisy_explanations[data_idx][model_idx]['LS'] = [LIMESegment(reshaper(x,j), trained_models[data_idx][model_idx], model_type=MODEL_TYPES[j], window_size=WINDOW_SIZES[i], cp=CPS[i]) for x in noisy_set]
        noisy_explanations[data_idx][model_idx]['N'] = [NEVES(reshaper(x,j), trained_models[data_idx][model_idx], datasets[data_idx][0], model_type=MODEL_TYPES[j]) for x in noisy_set]
        noisy_explanations[data_idx][model_idx]['LF'] = [LEFTIST(reshaper(x,j), trained_models[data_idx][model_idx], datasets[data_idx][0], model_type=MODEL_TYPES[j],) for x in noisy_set]
        
        j = j + 1
    i = i + 1

evaluation_metrics = {}
for data_idx in datasets.keys():
    evaluation_metrics[data_idx] = {}
    j = 0 
    for model_idx in trained_models[data_idx].keys():
        evaluation_metrics[data_idx][model_idx] = {}
        for explanation_idx in explanations[data_idx][model_idx].keys():
            evaluation_metrics[data_idx][model_idx][explanation_idx] = {}
            # Robustness
            evaluation_metrics[data_idx][model_idx][explanation_idx]['Robustness'] = robustness(explanations[data_idx][model_idx][explanation_idx],
                                                                                         noisy_explanations[data_idx][model_idx][explanation_idx])
            explanation_set = datasets[data_idx][2][0:10]
            explanation_labels = datasets[data_idx][3][0:10]
            if j == 0:
                explanation_predictions = trained_models[data_idx][model_idx].predict(explanation_set.reshape(explanation_set.shape[:2]))
            
            else:
                explanation_predictions = trained_models[data_idx][model_idx].predict(explanation_set)
                
                # Faithfulness
            evaluation_metrics[data_idx][model_idx][explanation_idx]['Faithfulness'] = faithfulness(explanations[data_idx][model_idx][explanation_idx],explanation_set,explanation_labels,explanation_predictions,trained_models[data_idx][model_idx],model_type=MODEL_TYPES[j])
        j+=1


print(evaluation_metrics)