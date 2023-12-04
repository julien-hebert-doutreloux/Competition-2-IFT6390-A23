import numpy as np
import pandas as pd
import os
def transform_labels(y_pred_A:list, y_pred_B:list):
    
    label_to_uppercase = lambda index: chr(index + 65)

    # Normalize ASCII sum
    def normalize_ascii_sum(ascii_sum):
        while ascii_sum > 122:
            ascii_sum -= 65
        return ascii_sum

    # Convert predictions to uppercase letters and then to ASCII
    ascii_predictions_a = [ord(label_to_uppercase(p)) for p in y_pred_A]
    ascii_predictions_b = [ord(label_to_uppercase(p)) for p in y_pred_B]

    # Sum and normalize ASCII values
    ascii_sums = [normalize_ascii_sum(a + b) for a, b in zip(ascii_predictions_a, ascii_predictions_b)]

    # Convert sums to characters
    transformed_labels = [chr(ascii_sum) for ascii_sum in ascii_sums]
    
    return transformed_labels
    
    
def export_prediction(transformed_labels, file_path, file_name):
    
    IDS = np.arange(len(transformed_labels))
    res = pd.DataFrame(data={"id":IDS, 'label':transformed_labels})
    
    prediction_path = os.path.join(file_path, f'{file_name}.csv')
    print(np.unique(res['label'], return_counts=True))
    res.to_csv(prediction_path, index=False)



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
    fig, ax_acc = plt.subplots(figsize=(16, 9))

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
                    ax_acc.plot(epochs, loaded_data, color='navy', alpha=0.3) # label=f'{data_type} - Training'
                else:
                    ax_acc.plot(epochs, loaded_data,color='crimson', alpha=0.3) #  label=f'{data_type} - Validation'

    ax_acc.tick_params(axis='x', labelsize='large')
    ax_acc.tick_params(axis='y', labelsize='large')
    ax_acc.set_title('Training & Validation Accuracy', fontsize=18)
    ax_acc.plot([], [], color='navy', label='Training')
    ax_acc.plot([], [], color='crimson', label='Validation')
    ax_acc.legend(loc='best', fontsize=14)
    ax_acc.set_xlabel("Epochs", fontsize=14)
    ax_acc.set_ylabel("Accuracy", fontsize=14)
    
    # Save the figures
    
    figure_path_acc = os.path.join('.', 'figures', 'combined_curves_accuracy.pdf')
    fig.savefig(figure_path_acc, bbox_inches='tight')
    plt.close()
    
    fig, ax_loss = plt.subplots(figsize=(16, 9))
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
                    ax_loss.plot(epochs, loaded_data, color='navy', alpha=0.3) #  label=f'{data_type} - Training'
                else:
                    ax_loss.plot(epochs, loaded_data, color='crimson', alpha=0.3) # label=f'{data_type} - Validation'

    # Set plot labels and title for loss
    ax_loss.tick_params(axis='x', labelsize='large')
    ax_loss.tick_params(axis='y', labelsize='large')
    ax_loss.set_title('Training & Validation Loss', fontsize=18)
    ax_loss.plot([], [], color='navy', label='Training')
    ax_loss.plot([], [], color='crimson', label='Validation')
    ax_loss.legend(loc='best',  fontsize=14)
    ax_loss.set_xlabel("Epochs",  fontsize=14)
    ax_loss.set_ylabel("Loss", fontsize=14)

    figure_path_loss = os.path.join('.', 'figures', 'combined_curves_loss.pdf')
    fig.savefig(figure_path_loss, bbox_inches='tight')
    plt.close()
    #plt.show()  # Optionally display the figures


    
