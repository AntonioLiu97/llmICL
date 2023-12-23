### Output the "out['logits'].cpu()" file and save to plot_output folder
### Result can be visualized using markov_learning_demo.ipynb

import os
os.environ['OMP_NUM_THREADS'] = '4'

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from utils.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data


# Check if directory exists, if not create it
if not os.path.exists('plot_output'):
    os.makedirs('plot_output')
    
llama_hypers = dict(
    alpha=0.99,
    beta=0.3,
    temp=float(0.5),
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep='', bit_sep='', minus_sign='-')
)


model_hypers = {
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    'llama-13b': {'model': 'llama-13b', **llama_hypers},
    'llama-70b': {'model': 'llama-70b', **llama_hypers},
    
}

model_predict_fns = {
    'llama-7b': get_llmtime_predictions_data,
    'llama-13b': get_llmtime_predictions_data,
    'llama-70b': get_llmtime_predictions_data,
}


model_names = list(model_predict_fns.keys())

# number of states
N_state = 10
# N_state = 4

states = np.arange(N_state)

# number of steps
Nt = 3000 


# Initialize the chain with a starting state, for example, state 0
chain = [0]

# Define the transition matrix P
# Rows sum to 1
def generate_transition_matrix(N_state):
    """
    Generate a random transition matrix of shape (N_state, N_state).
    Each row sums to 1.
    """
    P = np.random.rand(N_state, N_state)
    P /= P.sum(axis=1)[:, np.newaxis]
    return P

# Generate the chain
np.random.seed(1)

P = generate_transition_matrix(N_state)

for t in range(1, Nt):
    current_state = chain[-1]
    next_state = np.random.choice(states, p=P[current_state])
    chain.append(next_state)
    
# Create a time span
tspan = np.arange(Nt)

# Store the generated time-series in a pandas DataFrame
df = pd.DataFrame({'Time': tspan, 'X': chain})

# Split the data into training and testing sets
train = df['X'][0:int(Nt/5*4)]
test = df['X'][int(Nt/5*4):]

print("Transition matrix:")
for row in P:
    # plt.figure(figsize=(4, 1), dpi=100)
    # plt.bar(states, row)
    print(" ".join(f"{x:.2f}" for x in row))
    
settings = llama_hypers["settings"]  
train_str = "".join(str(x) for x in train.values)
test_str = "".join(str(x) for x in test.values)
full_series = "".join(str(x) for x in chain)
full_array = np.array(chain)

from models.llama import get_model_and_tokenizer
from jax import grad,vmap
import torch

# model, tokenizer = get_model_and_tokenizer('7b')
model, tokenizer = get_model_and_tokenizer('13b')
# model, tokenizer = get_model_and_tokenizer('70b')

batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True
    )
batch = {k: v.cuda() for k, v in batch.items()}

with torch.no_grad():
    out = model(**batch)
   
with open(f"plot_output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl", 'wb') as f:
            pickle.dump(out['logits'].cpu(), f)