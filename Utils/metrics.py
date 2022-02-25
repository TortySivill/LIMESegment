import numpy as np


def add_noise(ts):
    mu, sigma = 0, 0.1 # mean and standard deviation
    noise = np.random.normal(mu, sigma, ts.shape[0])
    noisy_ts = np.add(ts.reshape(ts.shape[0]),noise.reshape(ts.shape[0]))
    return noisy_ts

def robustness(explanations, noisy_explanations):
    robust = 0 
    for i in range(0,len(explanations)):
        print(explanations)
        original_order = np.argsort(explanations[i][0])
        noisy_order = np.argsort(noisy_explanations[i][0])
        if np.array_equal(original_order,noisy_order[:len(original_order)]):
            robust += 1
    return robust/len(explanations)

def reverse_segment(ts, index0, index1):
    perturbed_ts = ts.copy()
    perturbed_ts[index0:index1] = np.flip(ts[index0:index1])
    return perturbed_ts

def faithfulness(explanations, x_test, y_test, original_predictions, model, model_type):
    perturbed_samples = []
    for i in range(0,len(explanations)):
        top_index = np.argmax(np.abs(explanations[i][0]))
        segment_indices = explanations[i][1]+[-1]
        example_ts = x_test[i].copy()
        reversed_sample = reverse_segment(example_ts,segment_indices[top_index],segment_indices[top_index+1])
        perturbed_samples.append(reversed_sample)

    if model_type == 'proba':
        reversed_predictions = model.predict(np.asarray(perturbed_samples))
        correct_indexes = []
        differences = []
        for i in range(0,len(y_test)):
            if y_test[i] == np.argmax(reversed_predictions[i]):
                correct_indexes.append(i)
        for index in correct_indexes:
            prediction_index = int(np.argmax(original_predictions[index]))
            differences.append(np.abs(original_predictions[index][prediction_index] - reversed_predictions[index][prediction_index]))
        return np.mean(differences)
    else:
        
        reversed_samples = np.asarray(perturbed_samples)
        reversed_predictions = model.predict(reversed_samples.reshape(reversed_samples.shape[:2]))
        correct_indexes = []
        for i in range(0,len(original_predictions)):
            if original_predictions[i] == reversed_predictions[i]:
                correct_indexes.append(i)
        return len(correct_indexes)/len(original_predictions)