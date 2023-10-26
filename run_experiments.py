import os
os.environ['OMP_NUM_THREADS'] = '4'

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
# openai.api_key = "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"
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
from models.validation_likelihood_tuning import get_autotuned_predictions_data

llama_hypers = dict(
    alpha=0.99,
    beta=0.3,
    temp=float(0.5),
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=',', bit_sep='', minus_sign='-')
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=',', bit_sep='', minus_sign='-')
)

promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)
arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

model_hypers = {
    'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
    'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
    'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
    'PromptCast GPT-3': {'model': 'text-davinci-003', **promptcast_hypers},
    'ARIMA': arima_hypers,
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    'llama-13b': {'model': 'llama-13b', **llama_hypers},
    'llama-70b': {'model': 'llama-70b', **llama_hypers},
    
}

model_predict_fns = {
    # 'ARIMA': get_arima_predictions_data,
    # 'LLMTime GPT-3': get_llmtime_predictions_data,
    # 'LLMTime GPT-3.5': get_llmtime_predictions_data,
    # 'LLMTime GPT-4': get_llmtime_predictions_data,
    # 'PromptCast GPT-3': get_promptcast_predictions_data,
    # 'llama-7b': get_llmtime_predictions_data,
    # 'llama-13b': get_llmtime_predictions_data,
    'llama-70b': get_llmtime_predictions_data,
}


model_names = list(model_predict_fns.keys())

### Drift-Diffusion SDE

# Time discretization
Nt = 400 # number of steps
dt =  0.2 # time step
tspan = np.linspace(0, Nt*dt, Nt)

# Drift and diffusion parameters
a = 0.3  # Drift rate
sigma = 0.8  # Noise level

# Initialize the time series
X = np.zeros(Nt)
X[0] = 0  # Starting point

np.random.seed(1)
# Generate the drift-diffusion time series
for t in range(1, Nt):
    dW = np.sqrt(dt) * np.random.normal()  # Wiener process (Brownian motion)
    X[t] = X[t-1] + a * dt + sigma * dW

# Store the generated time-series in a pandas DataFrame
df = pd.DataFrame({'Time': tspan, 'X': X})

# Split the data into training and testing sets
train = df['X'][0:int(Nt/3*2)]
test = df['X'][int(Nt/3*2):]

### Temperature tuning
num_samples = 2
temp_list = [0.1, 0.5, 1, 1.5, 2.0]
# temp_list = [0.5, 1, 1.5]
# temp_list = [1.3]
### Llama 70-B takes 30 minutes to run

for model in model_names: # GPT-4 takes a about a minute to run
    hypers = list(grid_iter(model_hypers[model]))
    for temp in temp_list:
        hypers[0]['temp'] = temp
        pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
        with open(f"plot_output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl", 'wb') as f:
            pickle.dump((train, test, pred_dict, model), f)