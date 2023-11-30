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

sys.path.append('.')
from scripts.Data import *

# Modele ensembliste
def experience3():

    # modele 1 pour determiner les signes (R, U)
    m1_label_dict = {0:17, 1:20} # 2
    # modele 2 pour determiner les signes (A, E, M, N, S)
    m2_label_dict = {0:0, 1:4, 2:12, 3:13, 4:18} # 5
    # modele 3 pour determiner les restants des signes
    m3_label_dict = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 6:8, 7:10, 8:11, 9:14, 10:15, 11:16, 12:19, 13:21, 14:22, 15:23, 16:24} # 17
    
    
    # data import
    train_df = pd.read_csv("./data/raw/sign_mnist_train.csv")
    old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
    test_df = pd.read_csv("./data/raw/test.csv")
    
    #
    m1_train_df = train_df[train_df['label'].isin(m1_label_dict.values())].copy()
    m2_train_df = train_df[train_df['label'].isin(m2_label_dict.values())].copy()
    m3_train_df = train_df[train_df['label'].isin(m3_label_dict.values())].copy()
    #
    m1_val_df = old_test_df[old_test_df['label'].isin(m1_label_dict.values())].copy()
    m2_val_df = old_test_df[old_test_df['label'].isin(m2_label_dict.values())].copy()
    m3_val_df = old_test_df[old_test_df['label'].isin(m3_label_dict.values())].copy()
    
    # Data manipulation
    m1_y_train = m1_train_df['label']
    m2_y_train = m2_train_df['label']
    m3_y_train = m3_train_df['label']
    
    m1_y_val = m1_val_df['label']
    m2_y_val = m2_val_df['label']
    m3_y_val = m3_val_df['label']

    # label
    label_binarizer = LabelBinarizer()
    #
    m1_y_train = label_binarizer.fit_transform(m1_y_train)
    m2_y_train = label_binarizer.fit_transform(m2_y_train)
    m3_y_train = label_binarizer.fit_transform(m3_y_train)
    #
    m1_y_val = label_binarizer.fit_transform(m1_y_val)
    m2_y_val = label_binarizer.fit_transform(m2_y_val)
    m3_y_val = label_binarizer.fit_transform(m3_y_val)


    # Normalization
    m1_X_train = m1_train_df.drop(columns=['label']).values
    m2_X_train = m2_train_df.drop(columns=['label']).values
    m3_X_train = m3_train_df.drop(columns=['label']).values
    #
    m1_X_val = m1_val_df.drop(columns=['label']).values
    m2_X_val = m2_val_df.drop(columns=['label']).values
    m3_X_val = m3_val_df.drop(columns=['label']).values
    
    
    X_test = test_df.values/255
    
    # Reshape
    m1_X_train = m1_X_train.reshape(-1,28,28,1)
    m2_X_train = m2_X_train.reshape(-1,28,28,1)
    m3_X_train = m3_X_train.reshape(-1,28,28,1)
    #
    m1_X_val = m1_X_val.reshape(-1,28,28,1)
    m2_X_val = m2_X_val.reshape(-1,28,28,1)
    m3_X_val = m3_X_val.reshape(-1,28,28,1)
    
    
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
    
    # Modele 2
    m2_datagen = ImageDataGenerator(
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
    m2_datagen.fit(m2_X_train)
    
    # Modele 3
    m3_datagen = ImageDataGenerator(
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
    m3_datagen.fit(m3_X_train)
    
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

    m1_model = tf.keras.models.clone_model(model)
    m2_model = tf.keras.models.clone_model(model)
    m3_model = tf.keras.models.clone_model(model)
    
    m1_model.add(Dense(units = len(m1_label_dict)-1 , activation = 'softmax'))
    m2_model.add(Dense(units = len(m2_label_dict) , activation = 'softmax'))
    m3_model.add(Dense(units = len(m3_label_dict) , activation = 'softmax'))
    model.add(Dense(units = 24 , activation = 'softmax'))
    
    m1_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    m2_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    m3_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    
    epochs = 20
    m1_weights_path = os.path.join('.','data','weights',f'm1_model_weights_{epochs}.h5')
    m2_weights_path = os.path.join('.','data','weights',f'm2_model_weights_{epochs}.h5')
    m3_weights_path = os.path.join('.','data','weights',f'm3_model_weights_{epochs}.h5')
    
    if not os.path.exists(m1_weights_path):
        m1_history = m1_model.fit(
            m1_datagen.flow(m1_X_train, m1_y_train, batch_size = 128),
            epochs = epochs,
            validation_data=(m1_X_val, m1_y_val),
            callbacks=[learning_rate_reduction]
            )
        m1_model.save_weights(m1_weights_path)
    else:
        m1_model.load_weights(m1_weights_path)
        
    if not os.path.exists(m2_weights_path):
        m2_history = m2_model.fit(
            m2_datagen.flow(m2_X_train, m2_y_train, batch_size = 128),
            epochs = epochs,
            validation_data=(m2_X_val, m2_y_val),
            callbacks=[learning_rate_reduction]
            )
        m2_model.save_weights(m2_weights_path)
    else:
        m2_model.load_weights(m2_weights_path)
        
    
        
    if not os.path.exists(m3_weights_path):
        m3_history = m3_model.fit(
            m3_datagen.flow(m3_X_train, m3_y_train, batch_size = 128),
            epochs = epochs,
            validation_data=(m3_X_val, m3_y_val),
            callbacks=[learning_rate_reduction]
            )
        m3_model.save_weights(m3_weights_path)
    else:
        m3_model.load_weights(m3_weights_path)
        

    
    # Prediction par le modele general
    weights_path = os.path.join('.','data','weights',f'model_weights_{epochs}.h5')
    if not os.path.exists(weights_path):
        return print('First execute experience2.py')
    else:
        model.load_weights(weights_path)
        

    # Splitting columns A1 to A784 into a separate DataFrame
    df_A = test_df.filter(like='pixel_a')
    test_A = df_A.values.reshape(-1,28,28,1)
    y_pred_A = np.argmax(model.predict(test_A), axis=1)
    y_pred_A[y_pred_A >= 9] +=  1
    df_A['label'] = y_pred_A
    
    # Splitting columns B1 to B784 into a separate DataFrame
    df_B = test_df.filter(like='pixel_b')
    test_B = df_B.values.reshape(-1,28,28,1)
    y_pred_B = np.argmax(model.predict(test_B),axis=1)
    y_pred_B[y_pred_B >= 9] +=  1
    df_B['label'] = y_pred_B
    
    
    # pourrait être fait plus efficacement
    for df in [df_A, df_B]:
        ensembliste_pred = []
        
        for index, row in df.iterrows():
            x = row.drop('label').values
            x = x.reshape(-1, 28, 28, 1)
            y = row['label']
            
            if y in m1_label_dict.values():
                y_p =  0 if m1_model.predict(x,verbose=0)<= 0.5 else 1
                ensembliste_pred.append(m1_label_dict[y_p])
                
            elif y in m2_label_dict.values():
                y_p =  np.argmax(m2_model.predict(x,verbose=0))
                ensembliste_pred.append(m2_label_dict[y_p])
                
            elif y in m3_label_dict.values():
                y_p =  np.argmax(m3_model.predict(x,verbose=0))
                ensembliste_pred.append(m3_label_dict[y_p])
                
            else:
                return print('ERROR outside label')
        df['confirmed_label'] = ensembliste_pred
        
    transformed_labels = [transform_labels(label_A, label_B) for label_A, label_B in zip(df_A['confirmed_label'], df_B['confirmed_label'])]
    IDS = np.arange(len(transformed_labels))
    res = pd.DataFrame(data={"id":IDS, 'label':transformed_labels})

    prediction_path = os.path.join('data','prediction','cnn.csv')
    print(np.unique(res['label'], return_counts=True))
    res.to_csv(prediction_path, index=False)
    
experience3()
