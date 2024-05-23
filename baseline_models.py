import numpy as np
import torch

def calculate_Markov_unigram(full_series):
    '''
    This function calculates the probability distribution of the next state at each point in the series
    using a naive unigram model. The result is a NumPy array with dimensions (len(full_series), num_states),
    where each row represents the probability distribution of states at that point.

    Parameters:
    full_series (str): The series for which the PDF is to be calculated.

    Returns:
    numpy.ndarray: A 2D array where each row is the probability distribution of states at each point in the series.
    '''
    # Determine unique states and their total number
    unique_states = sorted(set(full_series))
    num_states = len(unique_states)
    
    # Create a mapping of state to index for array population
    state_to_index = {state: index for index, state in enumerate(unique_states)}

    # Initialize a matrix to hold the probability distributions
    # add 2 to be consistent with EOS
    probabilities_matrix = np.zeros((len(full_series)+2, num_states))

    # Initialize a dictionary to count occurrences of each state
    state_counts = {state: 0 for state in unique_states}

    for i, state in enumerate(full_series):
        # Update the state count and total states
        state_counts[state] += 1

        # Update the probabilities for each state up to the current point
        for state, count in state_counts.items():
            probabilities_matrix[i+2, state_to_index[state]] = count

    # convert un-normalized probabilities to logits
    probabilities_matrix = np.log(probabilities_matrix + 1e-10)
    probabilities_matrix = probabilities_matrix.reshape((1, len(full_series)+2, num_states))
    return torch.tensor(probabilities_matrix, dtype=torch.float32)

def calculate_Markov_bigram(full_series):
    '''
    This function calculates the probability distribution of the next state at each point in the series
    using a naive unigram model. The result is a NumPy array with dimensions (len(full_series), num_states),
    where each row represents the probability distribution of states at that point.

    Parameters:
    full_series (str): The series for which the PDF is to be calculated.

    Returns:
    numpy.ndarray: A 2D array where each row is the probability distribution of states at each point in the series.
    '''
    # Determine unique states and their total number
    unique_states = sorted(set(full_series))
    num_states = len(unique_states)
    
    # Create a mapping of state to index for array population
    state_to_index = {state: index for index, state in enumerate(unique_states)}

    # Initialize a matrix to hold the probability distributions
    # add 2 to be consistent with EOS
    # probabilities_matrix = np.zeros((len(full_series)+2, num_states)) + 1e-10
    probabilities_matrix = np.zeros((len(full_series)+2, num_states)) + 1

    # Un-normalized transition matrix
    P = np.zeros((num_states, num_states))

    for i in range(1, len(full_series)):
        pre_state = full_series[i-1]
        curr_state = full_series[i]
        # Update the state count and total states
        P[state_to_index[pre_state], state_to_index[curr_state]] += 1
        probabilities_matrix[i+2, :] += P[state_to_index[curr_state]]

    # convert un-normalized probabilities to logits
    probabilities_matrix = np.log(probabilities_matrix )
    probabilities_matrix = probabilities_matrix.reshape((1, len(full_series)+2, num_states))
    return torch.tensor(probabilities_matrix, dtype=torch.float32)


def AR1_with_sigma(time_series):
    time_series = np.asarray(time_series)
    mean_ts = np.mean(time_series)
    centered_ts = time_series - mean_ts
    
    phi = np.sum(centered_ts[1:] * np.roll(centered_ts, -1)[:-1]) / np.sum(np.roll(centered_ts, -1)[:-1] ** 2)
    
    # Calculate predicted values for the entire series based on phi
    predicted_values = np.empty_like(time_series)
    predicted_values[0] = time_series[0]  # Assuming the first value is as observed
    for t in range(1, len(time_series)):
        predicted_values[t] = phi * time_series[t-1] + mean_ts  # Adjusting each prediction by adding back the mean
    
    # Calculate residuals
    residuals = time_series - predicted_values
    
    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)
    
    # Predict the next state
    next_state = phi * time_series[-1] + mean_ts  # Adjusting the prediction by adding back the mean
    
    return next_state, residual_std

def AR1(time_series):
    predicted_mean_arr = [1]
    predicted_sigma_arr = [1]
    for i in range(1, len(time_series)):
        curr_series = time_series[:i]
        mean, sigma = AR1_with_sigma(curr_series)
        predicted_mean_arr += [mean]
        predicted_sigma_arr += [sigma]
    return np.array(predicted_mean_arr), np.array(predicted_sigma_arr)