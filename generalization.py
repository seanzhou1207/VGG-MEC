import numpy as np
import torch
from tqdm import tqdm

def count_correct_predictions(predicted, actual):
    # Element-wise comparison to check where predictions are correct
    correct_predictions = predicted == actual
    
    # Extract the classes from the actual where predictions were correct
    correct_classes = actual[correct_predictions]
    
    # Count the number of correct predictions for each unique class
    unique_classes, correct_counts = np.unique(correct_classes, return_counts=True)
    
    # Create a dictionary to map each class to its count of correct predictions
    correct_predictions_per_class = dict(zip(unique_classes, correct_counts))
    
    return correct_predictions_per_class

def get_predicted_and_actuals(model,data_loader,device):
    model.eval()
    predicted_all = []
    labels_all = []
    with torch.no_grad():
        for data,label in tqdm(data_loader,desc="Accuracy"):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _,predicted = torch.max(output.data,1)
            predicted_all.append(predicted)
            labels_all.append(label)
            # total += label.size(0)
            # correct += (predicted == label).sum().item()
            
    tensors_cpu = [t.cpu() for t in predicted_all]
    # Step 2: Concatenate all tensors
    concatenated_tensor = torch.cat(tensors_cpu)
    # Optional: Convert to NumPy array to strip all device info
    predicted_array = concatenated_tensor.numpy()
    
    tensors_cpu = [t.cpu() for t in labels_all]
    # Step 2: Concatenate all tensors
    concatenated_tensor = torch.cat(tensors_cpu)
    # Optional: Convert to NumPy array to strip all device info
    labels_array = concatenated_tensor.numpy()

    #print(predicted_array[:10])
    #print(labels_array[:10])
    #class_predicted = count_correct_predictions(predicted_array, labels_array)
    #print(class_predicted)
    #ks = [class_predicted[i] for i in class_predicted.keys()]
    return predicted_array, labels_array

def get_class_probabilities(labels):
    # Count the occurrences of each unique class
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Calculate probabilities by dividing counts by the total number of labels
    probabilities = counts / len(labels)
    
    # Combine class labels with their corresponding probabilities
    class_probabilities = dict(zip(unique_classes, probabilities))
    
    return class_probabilities