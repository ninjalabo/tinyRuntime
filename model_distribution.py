# Don't edit this file! This was automatically generated from "model_distribution.ipynb".

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def get_channel_data(conv_layer):
    """ this is designed for convolutional layers as the plots and data extraction take the channel dimension into account """
    weights = conv_layer.weight.data
    num_channels = weights.shape[0]
    channel_data = [weights[i].cpu().numpy().flatten() for i in range(num_channels)]

    return channel_data

def get_variance(data):
    """ compute the iqr values, the boxplot does these when plotting as well """
    iqr_values = []
    lower_boundaries = []
    upper_boundaries = []

    for channel_weights in data:
        # calculate the quartiles
        Q1 = np.percentile(channel_weights, 25)
        Q3 = np.percentile(channel_weights, 75)
        iqr = Q3 - Q1
        
        # calculate the whisker boundaries
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr
        
        iqr_values.append(iqr)
        lower_boundaries.append(lower_bound)
        upper_boundaries.append(upper_bound)
    iqr_values = np.array(iqr_values)
    lower_boundaries = np.array(lower_boundaries)
    upper_boundaries = np.array(upper_boundaries)

    # calculate the variance of the lower and upper boundaries
    lower_boundary_variance = np.var(lower_boundaries)
    upper_boundary_variance = np.var(upper_boundaries)

    return lower_boundary_variance, upper_boundary_variance
    

def get_distribution(weights):
    """ compute the skewness and kurtosis values """
    mean = np.mean(weights)
    std = np.std(weights)
    skewness = skew(weights)
    kurt = kurtosis(weights)
    
    return mean, std, skewness, kurt

def plot_histogram(weights):
    """ plot histogram of the weights """
    plt.hist(weights, bins=50, color='blue', alpha=0.7)
    plt.title('Weight Distribution Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.show()

def plot_weight_boxplot(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True)
    
        
    plt.xlabel('Output Channel Index')
    plt.xticks(range(1, len(data) + 1), rotation=45, fontsize=8)
    plt.ylabel('Weight Range')
    plt.title('Original Weight Ranges')
    plt.grid(True)
    plt.show()

def plot_layers(model, visualize=False):
  conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]

  for layer in conv_layers:
      data = get_channel_data(layer)
      lower_var, upper_var = get_variance(data)
      all_weights = np.array(data).flatten()
      mean, std, skewness, kurt = get_distribution(all_weights)
      print(f'Layer: {layer}')
      print(f'Lower Bound Variance: {lower_var}')
      print(f'Upper Bound Variance: {upper_var}')
      print(f'Mean: {mean}')
      print(f'Standard Deviation: {std}')
      # kurtosis indicates the heaviness of the tails of the distribution, more outliers with higher kurtosis
      # skewness indicates the asymmetry of the distribution, positive right skewed, negative left skewed
      # low skewness and kurtosis values are good for per layer quantization
      
      print(f'Skewness: {skewness}')
      print(f'Kurtosis: {kurt}')
      print('---------------------------------')
      if visualize:
          #plot_histogram(all_weights)
          plot_weight_boxplot(data)
