import torch.nn as nn

from torchvision import datasets, transforms

import numpy as np

def get_mec_data(img_tensor):
    """
    Calculate the information capacity of a dataset with black and white images
    """
    n_img, n_row, n_col = img_tensor.shape

    info = n_img * n_row * n_col    # Assume 1 bit per pixel

    return info

def get_mec_ffnn(layer_ffnn: nn.Sequential,dropout=0.5):
    """
    Get MEC of last feed forward neural net layer
    """
    mec_layer = []
    print("The linear layers are: ")
    print("------------------------------------------")
    for layer in layer_ffnn:
        if isinstance(layer, nn.Linear):
            print(layer)
            dim_in = layer.in_features
            dim_out = layer.out_features
            print(dim_in, dim_out)
            #if layer.bias:
            if mec_layer == []:
                mec = (dim_in + 1) * dim_out
                mec_layer.append(mec)
            else: 
                #print((dim_in + 1) * dim_out)
                mec = min(int(dim_in*dropout), (dim_in + 1) * dim_out)
                mec_layer.append(mec)    
    print("------------------------------------------")
    mec_total = np.sum(mec_layer)

    return mec_total

def get_compression(dim_in, dim_out):
    """
    Get compression ratio of convolution layer
    """
    comp_ratio = round(dim_in/dim_out, 2)

    print(f"Compression ratio of convolution layer: {comp_ratio} : 1")
    return comp_ratio

# GPT Code
def get_class_probabilities(labels):
    # Count the occurrences of each unique class
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Calculate probabilities by dividing counts by the total number of labels
    probabilities = counts / len(labels)
    
    # Combine class labels with their corresponding probabilities
    class_probabilities = dict(zip(unique_classes, probabilities))
    
    return class_probabilities

def get_G(mec, ks, ps):
    """
    Calculate generalization for model
    """
    numerator = -np.sum(np.log2(ps)*ks)
    return numerator/mec