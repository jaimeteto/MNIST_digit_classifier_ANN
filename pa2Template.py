#    name: pa2Template.py
# purpose: template for building a Keras model
#          for hand written number classification
#    NOTE: Submit a different python file for each model
# -------------------------------------------------


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import utils as np_utils
from keras.models import load_model

from pa2pre import processTestData
import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()

def main():
    np.random.seed(1671)

    parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    (X_train, y_train) = processTestData(X_train,y_train)

    print('KERA modeling build starting...')
    ## Build your model here


    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
    
    
    
    
    # Part 2 - Building the ANN

    # # Initializing the ANN
    ann = tf.keras.models.Sequential()

    # # Adding the first hidden layer

    ann.add(tf.keras.layers.Dense(units=500, activation='relu',input_shape=(784,)))

    # # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=500, activation='relu'))

    # # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

    # Part 3 - Training the ANN

    # Compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training the ANN on the Training set
    r = ann.fit(X_train, y_train, batch_size = 32, epochs = 1)

    # Part 4 - Making the predictions and evaluating the model


    # Predicting the Test set results
    y_pred = ann.predict(X_test)
    #print(np.concatenate((y_pred.reshape(len(y_pred)), y_test.reshape(len(y_test),1)),1))

    print(y_pred[0])
    print(y_test[0])

    # Making the Confusion Matrix
    from sklearn.metrics import accuracy_score
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # accuracy_score(y_test, y_pred)

    y_pred_labels = [np.argmax(i) for i in y_pred]
    y_test_labels = [np.argmax(i)for i in y_test]
    cm = tf.math.confusion_matrix(labels=y_test_labels, predictions= y_pred_labels)
    print(cm)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(accuracy)


    #saving model
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # heat map


if __name__ == '__main__':
    main()
