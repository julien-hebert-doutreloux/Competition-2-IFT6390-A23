import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator

# https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy

# Import from the project root
train_df = pd.read_csv("./data/raw/sign_mnist_train.csv")
x_test = pd.read_csv("./data/raw/test.csv") # no label

# Label preprocess
y_train = train_df['label']
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
del train_df['label']

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = train_df.values
x_train = x_train/255
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test/255
x_test = test.reshape(-1,28,28,1)


# data augmentation
datagen = ImageDataGenerator(
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
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

# Function to generate augmented images and store them in a DataFrame (flattened)
def generate_augmented_flattened_images(generator, data, labels=None):
    augmented_images = []
    augmented_labels = [] if labels is not None else None

    # Generate augmented images
    for x_batch, y_batch in generator.flow(data, labels, batch_size=128, shuffle=False):
        augmented_images.append(x_batch[0].flatten())

        if labels is not None:
            augmented_labels.append(np.argmax(y_batch))

        if len(augmented_images) >= 10*len(data):
            break

    # Convert the list of augmented flattened images to a numpy array
    augmented_images = np.array(augmented_images)

    # Create a DataFrame to store augmented flattened images
    augmented_df = pd.DataFrame(augmented_images)

    if labels is not None:
        augmented_df['label'] = augmented_labels

    return augmented_df

# Generate augmented flattened images from the data
augmented_flattened_df = generate_augmented_flattened_images(datagen, x_train, y_train)

# Display the augmented flattened images DataFrame
print(augmented_flattened_df.head())

table = pa.Table.from_pandas(augmented_flattened_df)
data_path = os.path.join('.','data','processed','augmented_flattened_images.pqr')
pq.write_table(table, data_path)












