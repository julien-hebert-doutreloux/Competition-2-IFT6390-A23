import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle

sys.path.append('.')
from scripts.Data import *


import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process random_state_int and weight_file_path.')
    parser.add_argument('--random_state_int', type=int, help='Random state integer')
    parser.add_argument('--weight_file_path', type=str, help='Path to the weight file')

    args = parser.parse_args()

    random_state = int(args.random_state_int) if args.random_state_int else 42
    weights_path = args.weight_file_path

    # data import
    old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
    test_df = pd.read_csv("./data/raw/test.csv")
    train_df = pd.read_csv("./data/raw/sign_mnist_train.csv").sample(frac=1)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=random_state)

    # Data manipulation
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = old_test_df['label']

    # label
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_val = label_binarizer.fit_transform(y_val)

    # Normalization
    X_train = train_df.drop(columns=['label']).values
    X_val = val_df.drop(columns=['label']).values/255
    X_test = old_test_df.drop(columns=['label']).values/255

    # Reshape
    X_train = X_train.reshape(-1,28,28,1)
    X_val = X_val.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)

    # Data augmentation
    # Modele 1
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images  
        brightness_range=[0.15,.65])
    datagen.fit(X_train)


    # Modele de base
    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 5, verbose=1, factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


    model = tf.keras.models.clone_model(model)
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    epochs = 1000

    if not weights_path:
        history = model.fit(
                    datagen.flow(X_train, y_train, batch_size = 128),
                    epochs = epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[learning_rate_reduction, early_stopping]
                    )
                    
        
        n_epochs = len(history.history['loss'])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_pred[y_pred >= 9] +=  1

        print(classification_report(y_test, y_pred))
        acc = np.mean(y_test==y_pred)
        print(f"Accuracy : {acc}")
        
        formatted_number = "{:.4f}".format(acc).replace('.','_')
        name_tag = f'{random_state}_{n_epochs}_{formatted_number}'
        weights_path = os.path.join('.','data','weights',f'model_weights_{name_tag}.h5')
        model.save_weights(weights_path)
        
        
        epochs = [i for i in range(n_epochs)]
        
        fig , ax = plt.subplots(1,2)
        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        

        # Define file paths for pickle files
        file_paths = {
            'train_loss': f'./data/asset/train_loss/train_loss_{name_tag}.pkl',
            'train_acc': f'./data/asset/train_acc/train_acc_{name_tag}.pkl',
            'val_loss': f'./data/asset/val_loss/val_loss_{name_tag}.pkl',
            'val_acc': f'./data/asset/val_acc/val_acc_{name_tag}.pkl'
        }

        # Save data to respective pickle files
        for key, data in zip(file_paths.keys(), [train_loss, train_acc, val_loss, val_acc]):
            with open(file_paths[key], 'wb') as file:
                pickle.dump(data, file)
        
        
        
        # fig.set_size_inches(16,9)

        # ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
        # ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
        # ax[0].set_title('Training & Validation Accuracy')
        # ax[0].legend()
        # ax[0].set_xlabel("Epochs")
        # ax[0].set_ylabel("Accuracy")

        # ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
        # ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
        # ax[1].set_title('Testing Accuracy & Loss')
        # ax[1].legend()
        # ax[1].set_xlabel("Epochs")
        # ax[1].set_ylabel("Loss")
        
        # figure_path = os.path.join('.','figures',f'cnn_ensembliste_{name_tag}.png')
        # plt.savefig(figure_path)

        
    else:
        model.load_weights(weights_path)
        name_tag = weights_path.split('model_weights_')[-1].replace('.h5', '')

        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_pred[y_pred >= 9] +=  1
        acc = np.mean(y_test==y_pred)
        print(classification_report(y_test, y_pred))
        print(f"Accuracy : {acc}")


    df_A = test_df.filter(like='pixel_a')
    test_A = df_A.values.reshape(-1,28,28,1)
    
    df_B = test_df.filter(like='pixel_b')
    test_B = df_B.values.reshape(-1,28,28,1)

    # Notre competition
    y_pred_A = np.argmax(model.predict(test_A), axis=1)
    y_pred_A[y_pred_A >= 9] +=  1
    
    y_pred_B = np.argmax(model.predict(test_B),axis=1)
    y_pred_B[y_pred_B >= 9] +=  1
    

    
    # ascii sum
    y_pred = transform_labels(y_pred_A,y_pred_B)
    
    # export prediction
    file_path = os.path.join('.','data','prediction')
    file_name = f'cnn_{name_tag}'
    export_prediction(y_pred, file_path, file_name)
