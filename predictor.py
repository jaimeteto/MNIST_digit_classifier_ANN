# load the model from disk
NB_CLASSES=10
import numpy as np
from keras import utils as np_utils
from sklearn import preprocessing
import pickle

def process_truth_file(input_file, output_file):
    odd_count = 0

    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            number = int(line.strip())
            label = 1 if number % 2 == 1 else 0

            # Count odd numbers
            if label == 1:
                odd_count += 1

            # Write to the output file
            output_f.write(f'{label}\n')

    return odd_count






#upload validation file
data = np.load('MNIST_autolabTest_X.npy')

#prepocessing data
X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

print(len(X))

#     # y preprocessing goes here.  y_test becomes a ohe
# y_ohe = np_utils.to_categorical (y, NB_CLASSES)


loaded_model = pickle.load(open("./model4.sav", 'rb'))

result = loaded_model.predict(X)
y_pred_labels = [np.argmax(i) for i in result]
print(y_pred_labels)
# Define the name of the text file
text_file_name = 'outputModule4.txt'

# Open the text file in write mode
with open(text_file_name, 'w') as file:
    # Iterate through the array
    for number in y_pred_labels:
        # Write each number to a new line in the text file
        file.write(str(number) + '\n')

print(f"Numbers have been written to {text_file_name}.")

input_path = 'outputModule4.txt'
output_path = 'truth-odd-even.txt'
# Call the function with the file paths
odd_count = process_truth_file(input_path, output_path)

# Output the count of odd numbers
print(f'Number of odd numbers: {odd_count}')
