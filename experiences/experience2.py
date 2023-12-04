import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.callbacks import EarlyStopping

sys.path.append('.')
from scripts.Data import *

# Modele ensembliste
def experience2():

    # data import
    train_df = pd.read_csv("./data/raw/sign_mnist_train.csv")
    old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
    
    test_df = pd.read_csv("./data/raw/test.csv")
    
    
    m1_train_df = train_df.copy()
    m1_val_df = old_test_df.copy()
    
    # Data manipulation
    m1_y_train = m1_train_df['label']
    m1_y_val = m1_val_df['label']


    # label
    label_binarizer = LabelBinarizer()
    #
    m1_y_train = label_binarizer.fit_transform(m1_y_train)
    m1_y_val = label_binarizer.fit_transform(m1_y_val)

    # Normalization
    m1_X_train = m1_train_df.drop(columns=['label']).values
    m1_X_val = m1_val_df.drop(columns=['label']).values
    
    # Reshape
    m1_X_train = m1_X_train.reshape(-1,28,28,1)
    m1_X_val = m1_X_val.reshape(-1,28,28,1)

    

    # Data augmentation
    # Modele 1
    m1_datagen = ImageDataGenerator(
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
    m1_datagen.fit(m1_X_train)
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1, factor=0.5, min_lr=0.00001)
    
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
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    m1_model = tf.keras.models.clone_model(model)
    m1_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    
    epochs = 100
    weights_path = os.path.join('.','data','weights',f'model_weights_{epochs}.h5')
    if not os.path.exists(weights_path):
        m1_history = m1_model.fit(
                    m1_datagen.flow(m1_X_train, m1_y_train, batch_size = 128),
                    epochs = epochs,
                    validation_data=(m1_X_val, m1_y_val),
                    callbacks=[learning_rate_reduction, early_stopping]
                    )
        m1_model.save_weights(weights_path)
    else:
        m1_model.load_weights(weights_path)
        
        
     # Splitting columns A1 to A784 into a separate DataFrame
    df_A = test_df.filter(like='pixel_a')
    test_A = df_A.values.reshape(-1,28,28,1)
    y_pred_A = np.argmax(m1_model.predict(test_A), axis=1)
    y_pred_A[y_pred_A >= 9] +=  1
    
    # Splitting columns B1 to B784 into a separate DataFrame
    df_B = test_df.filter(like='pixel_b')
    test_B = df_B.values.reshape(-1,28,28,1)
    y_pred_B = np.argmax(m1_model.predict(test_B),axis=1)
    y_pred_B[y_pred_B >= 9] +=  1    
    
    # ascii sum
    y_pred = transform_labels(y_pred_A,y_pred_B)
    
    # export prediction
    file_path = os.path.join('.','data','prediction')
    file_name = f'cnn_1'
    export_prediction(y_pred, file_path, file_name)

experience2()
