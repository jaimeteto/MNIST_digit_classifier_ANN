#    name: pa2Template2.py
# purpose: template for building a Keras model
#          for hand written number classification using L2 regularization
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
import pickle
from keras import regularizers
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


    output_file = parms.outModelFile

    (X_train, y_train) = processTestData(X_train,y_train)

    print('KERA modeling build starting...')
    ## Build your model here


    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
    
    #create batches
    # Convert input data and labels into a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    #batch size
    batch_size = 32
    #number of epeochs
    num_epochs = 250
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
        
    
    # Part 2 - Building the ANN

    # # Initializing the ANN
    ann = tf.keras.models.Sequential()

    # # Adding the first hidden layer

    ann.add(tf.keras.layers.Dense(units=500, activation='relu',input_shape=(784,),kernel_regularizer=regularizers.l2(0.01)))

    # # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=500, activation='relu',kernel_regularizer=regularizers.l2(0.1)))

    # # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

    # Part 3 - Training the ANN

    # Compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training the ANN on the Training set
    #r = ann.fit(X_train, y_train, batch_size = 32, epochs = 1)
    # Train the model
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for batch_input, batch_labels in dataset:
            ann.train_on_batch(batch_input, batch_labels)

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
    pickle.dump(ann, open(output_file, 'wb'))

    # heat map
    import matplotlib.pylab as plt
    import seaborn as sns
    ideal_matrix = np.array([[0.9,0.1,0.1], [0.1,0.9,0], [0,0,1]])
    sns.set()

    ax = sns.heatmap(cm, annot=True, fmt='.1f',linewidth=0.5)
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')

    #plt.savefig("hm.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf
    plt.show()


if __name__ == '__main__':
    main()
