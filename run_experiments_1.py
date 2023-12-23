### BM, prec = 3, rescale_factor = 0.7, up_shift = 0.15, Llama 13b, , refine_depth = -2, mode = "all"
import os
os.environ['OMP_NUM_THREADS'] = '4'


import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from utils.serialize import serialize_arr, deserialize_str, SerializerSettings
from ipywidgets import interact
import numpy as np
import torch
from utils.ICL import MultiResolutionPDF, recursive_refiner, trim_kv_cache
from scipy.stats import norm

import pickle
import os
if not os.path.exists('MultiPDFList'):
    os.makedirs('MultiPDFList')

# Check if directory exists, if not create it
if not os.path.exists('plot_output'):
    ### dump pickled torch files here for later plotting
    os.makedirs('plot_output')
    
if not os.path.exists('gif_plots'):
    ### dump generated GIFs
    os.makedirs('gif_plots')

### Drift-Diffusion SDE

# Time discretization
# Nt = 1000 # number of steps
Nt = 700
# Nt = 500 # number of steps
# Nt = 300 # number of steps
# Nt = 10 # number of steps
dt =  0.2 # time step
tspan = np.linspace(0, Nt*dt, Nt)

# Drift and diffusion parameters
# Drift rate
np.random.seed(6)
# a = 0.3  
a = 0
# Noise level
sigma = 0.8

# Initialize the time series
x = 0  # Starting point
time_series = [x]
mean_series = [x]
sigma_series = [0]

# Generate the drift-diffusion time series
for t in range(1, Nt):
    x_mean = x + a*dt
    x_sigma = sigma * np.sqrt(dt)
    dW =  np.random.normal()  # Wiener process (Brownian motion)
    x = x_mean + x_sigma * dW
    
    time_series.append(x)
    mean_series.append(x_mean)
    sigma_series.append(x_sigma)
    
# Store the generated time-series in a pandas DataFrame
df = pd.DataFrame({'X': time_series})

# Split the data into training and testing sets
train = df['X'][0:int(Nt/3*2)]
test = df['X'][int(Nt/3*2):]

prec = 3
settings=SerializerSettings(base=10, prec=prec, signed=True, time_sep=',', bit_sep='', minus_sign='-', fixed_length=True, max_val = 10)

from utils.serialize import serialize_arr
X = np.append(train.values, test.values) 
rescale_factor = 0.7
up_shift = 0.15

rescaled_array = (X-X.min())/(X.max()-X.min()) * rescale_factor + up_shift
rescaled_true_mean_arr = (np.array(mean_series)-X.min())/(X.max()-X.min()) * rescale_factor + up_shift
rescaled_true_sigma_arr = np.array(sigma_series)/(X.max()-X.min()) * rescale_factor 

full_series = serialize_arr(rescaled_array, settings)
full_array = deserialize_str(full_series, settings) * 10

full_series = full_series.lstrip('0').replace(',0', ',')

import torch
torch.cuda.empty_cache()
from models.llama import get_model_and_tokenizer, get_tokenizer

model, tokenizer = get_model_and_tokenizer('13b')

batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True        
    )

torch.cuda.empty_cache()
with torch.no_grad():
    out = model(batch['input_ids'].cuda(), use_cache=True)
    # out = model(batch['input_ids'].cpu(), use_cache=True)

logit_mat = out['logits'] 
kv_cache_main = out['past_key_values']

good_tokens_str = list("0123456789")
good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
logit_mat_good = logit_mat[:,:,good_tokens].clone()

#### refine_depth = -2
### 2min49 ~ 2min57 sec without cache
### 26 sec with cache

#### refine_depth = -1
### with cache

### 21 mins for 300 steps, depth = -1, curr = -3
T = 1
probs = torch.nn.functional.softmax(logit_mat_good[:,1:,:] / T, dim=-1)

PDF_list = []
comma_locations = np.sort(np.where(np.array(list(full_series)) == ',')[0])

for i in range(len(comma_locations)):
# for i in range(5):
    # print(i)
    PDF = MultiResolutionPDF()
    # slice out the number before ith comma
    if i == 0:
        start_idx = 0
    else:
        start_idx = comma_locations[i-1]+1
    end_idx = comma_locations[i]
    num_slice = full_series[start_idx:end_idx]
    prob_slice = probs[0,start_idx:end_idx].cpu().numpy()
    ### Load hierarchical PDF 
    PDF.load_from_num_prob(num_slice, prob_slice)
    
    ### Refine hierarchical PDF
    seq = full_series[:end_idx]
    # cache and full_series are shifted from beginning, not end
    end_idx_neg = end_idx - len(full_series)
    # kv cache contains seq[0:-1]
    kv_cache = trim_kv_cache(kv_cache_main, end_idx_neg-1)
    recursive_refiner(PDF, seq, curr = -prec, main = True, refine_depth = -2, mode = "all", 
                      kv_cache = kv_cache, model = model, tokenizer = tokenizer, good_tokens=good_tokens)
    
    PDF_list += [PDF]
    
        
### Pickle PDF_list

with open(f"MultiPDFList/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl", 'wb') as f:
    pickle.dump(PDF_list, f)
    
