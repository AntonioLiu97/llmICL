import numpy as np
import torch

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  # Third layer
        self.fc4 = nn.Linear(hidden_size3, output_size)  # Adjusted final layer
        self.activation = nn.LeakyReLU()  # Shared activation function
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        output = self.fc4(x)  # Note: No activation after the last layer if it's meant for a classification output layer
        return output

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
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

def non_linear_AR1_with_sigma(time_series, p=2):
    time_series = np.asarray(time_series)
    mean_ts = np.mean(time_series)
    centered_ts = time_series - mean_ts

    # Extend the basis to include power terms up to p and a constant term
    power_bases = [centered_ts**i for i in range(1, p+1)]
    ones = np.ones_like(centered_ts[:-1])  # Constant term
    combined_ts = np.vstack([ones] + [base[:-1] for base in power_bases]).T
    
    # Prepare the target vector for the linear system
    target = centered_ts[1:]
    
    # Use np.linalg.lstsq to solve the least squares of AX = B
    coefficients, residuals, rank, s = np.linalg.lstsq(combined_ts, target, rcond=None)
    
    # Calculate predicted values for the entire series based on coefficients
    predicted_values = np.empty_like(time_series)
    predicted_values[0] = time_series[0]  # Assuming the first value is as observed
    for t in range(1, len(time_series)):
        predicted_value = coefficients[0]  # Start with the constant term
        for i in range(1, len(coefficients)):
            predicted_value += coefficients[i] * (centered_ts[t-1] ** i)
        predicted_values[t] = predicted_value + mean_ts
    
    # Calculate residuals
    residuals = time_series - predicted_values
    
    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)
    
    # Predict the next state using the power bases
    next_state = coefficients[0]  # Start with the constant term
    for i in range(1, len(coefficients)):
        next_state += coefficients[i] * (centered_ts[-1] ** i)
    next_state += mean_ts
    
    return next_state, residual_std

def non_linear_AR1(time_series, p=2):
    predicted_mean_arr = [0.5]
    predicted_sigma_arr = [0.5]
    for i in range(1, len(time_series)):
        curr_series = time_series[:i]
        mean, sigma = non_linear_AR1_with_sigma(curr_series, p)
        predicted_mean_arr += [mean]
        predicted_sigma_arr += [sigma]
    return np.array(predicted_mean_arr), np.array(predicted_sigma_arr)


def NN_AR1_with_sigma(model, time_series, epochs=1000, lr=0.001, patience=30, fix_sigma = False):
    # time_series = np.asarray(time_series, dtype=np.float32)
    X_tensor = time_series[:-1].reshape(-1, 1)
    Y_tensor = time_series[1:].reshape(-1)  
    print(X_tensor.shape)
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_tensor)
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = torch.exp(log_sigma)
        # Negative log likelihood
        if fix_sigma:
            loss = torch.mean((Y_tensor - mu) ** 2) 
        else:    
            loss = torch.mean(log_sigma + 0.5 * ((Y_tensor - mu) ** 2) / sigma**2)

        loss.backward()
        optimizer.step()
        
        # Check for early stopping
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                # print(f'Early stop at epoch {epoch+1}')
                break
            
        # if (epoch+1) % 10 == 0:            
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    print(f'Final epoch: {epoch+1}, NLL: {loss.item():.4f}')
    # Predict the next state
    model.eval()
    with torch.no_grad():
        last_x = time_series[-1].reshape(-1, 1)
        pred = model(last_x)
        mu, log_sigma = pred[0][0].item(), pred[0][1].item()
        sigma = np.exp(log_sigma)
    
    if fix_sigma:
        mu, 0.1
    else:
        return mu, sigma

def NN_AR1(time_series, fix_sigma = False):
    predicted_mean_arr = [0.5]
    predicted_sigma_arr = [0.5]
    # Define the model
    model = SimpleNN(input_size=1, hidden_size1=64, hidden_size2=32, hidden_size3=16, output_size=2).to(device)
    time_series = torch.tensor(time_series, dtype=torch.float32).to(device)
    for i in tqdm(range(1, len(time_series))):
    # for i in tqdm(range(10, 11)):
        # model = SimpleNN(input_size=1, hidden_size1=64, hidden_size2=32, hidden_size3=16, output_size=2).to(device)
        curr_series = time_series[:i]
        mean, sigma = NN_AR1_with_sigma(model, curr_series, epochs=1000, lr=0.001, patience=30, fix_sigma = fix_sigma)
        predicted_mean_arr += [mean]
        predicted_sigma_arr += [sigma]
    return np.array(predicted_mean_arr), np.array(predicted_sigma_arr)

def AR1_with_sigma(time_series):
    time_series = np.asarray(time_series)
    mean_ts = np.mean(time_series)
    centered_ts = time_series - mean_ts

    # Extend the basis to include quadratic terms and a constant term
    ones = np.ones_like(centered_ts[:-1])  # Constant term
    combined_ts = np.vstack([ones, centered_ts[:-1]]).T
    
    # Prepare the target vector for the linear system
    target = centered_ts[1:]
    

    # Use np.linalg.lstsq to solve the least squares of AX = B
    coefficients, residuals, rank, s = np.linalg.lstsq(combined_ts, target, rcond=None)
    phi_0, phi_linear = coefficients


    # Calculate predicted values for the entire series based on phi_0, phi_linear, and phi_quadratic
    predicted_values = np.empty_like(time_series)
    predicted_values[0] = time_series[0]  # Assuming the first value is as observed
    for t in range(1, len(time_series)):
        predicted_values[t] = phi_0 + phi_linear * centered_ts[t-1] + mean_ts
    
    # Calculate residuals
    residuals = time_series - predicted_values
    
    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)
    
    # Predict the next state using constant, linear, and quadratic terms
    next_state = phi_0 + phi_linear * centered_ts[-1] + mean_ts
    
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

