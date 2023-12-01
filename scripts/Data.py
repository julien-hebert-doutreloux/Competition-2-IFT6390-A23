import numpy as np
import pandas as pd

def transform_labels(label_A, label_B):
    # Compute the ASCII sum of the labels and add them together
    capital_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    ascii_sum = ord(capital_alphabet[label_A]) + ord(capital_alphabet[label_B])

    # Adjust the sum if it exceeds the range 122 (ASCII for 'z')
    while ascii_sum > 122:
        ascii_sum -= 65
    # Convert the ASCII sum to its corresponding character
    final_char = chr(ascii_sum)

    # Convert the character to string dtype
    return str(final_char)
    
    
def train_vs_val():
    import os
    import pickle
    import glob
    import matplotlib.pyplot as plt

    # Define the directory path
    directory_path = './data/asset/'

    # Define the patterns to match files
    file_patterns = {
        'train_loss': 'train_loss/train_loss_*.pkl',
        'train_acc': 'train_acc/train_acc_*.pkl',
        'val_loss': 'val_loss/val_loss_*.pkl',
        'val_acc': 'val_acc/val_acc_*.pkl'
    }

    # Create separate figures for accuracy and loss
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(16, 11))

    # Separate curves for accuracy
    for data_type, file_pattern in file_patterns.items():
        if 'acc' in data_type:
            # Search for files matching the pattern in the directory
            for file_path in glob.glob(os.path.join(directory_path, file_pattern)):
                # Load data from pickle files
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)

                # Generate curve based on the loaded data
                epochs = range(len(loaded_data))  # Assuming epochs data is available
                if 'train' in data_type:
                    ax_acc.plot(epochs, loaded_data, label=f'{data_type} - Training')
                else:
                    ax_acc.plot(epochs, loaded_data, label=f'{data_type} - Validation')

    # Set plot labels and title for accuracy
    ax_acc.set_title('Training & Validation Accuracy')
    #ax_acc.legend()
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")

    # Separate curves for loss
    for data_type, file_pattern in file_patterns.items():
        if 'loss' in data_type:
            # Search for files matching the pattern in the directory
            for file_path in glob.glob(os.path.join(directory_path, file_pattern)):
                # Load data from pickle files
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)

                # Generate curve based on the loaded data
                epochs = range(len(loaded_data))  # Assuming epochs data is available
                if 'train' in data_type:
                    ax_loss.plot(epochs, loaded_data, label=f'{data_type} - Training')
                else:
                    ax_loss.plot(epochs, loaded_data, label=f'{data_type} - Validation')

    # Set plot labels and title for loss
    ax_loss.set_title('Training & Validation Loss')
    #ax_loss.legend()
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")

    # Save the figures
    figure_path_acc = os.path.join('.', 'figures', 'combined_curves_accuracy.png')
    fig.savefig(figure_path_acc, bbox_inches='tight')

    figure_path_loss = os.path.join('.', 'figures', 'combined_curves_loss.png')
    fig.savefig(figure_path_loss, bbox_inches='tight')

    #plt.show()  # Optionally display the figures


    
