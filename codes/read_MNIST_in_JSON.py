"""Converts MNIST dataset into JSON objects

Remember that JSON object will be considerably larger than the source.
So, it might be better to use the default files directly.
"""
#!/usr/bin/env python
import random
import struct
import json
import numpy as np


def read(dtype):
    """ Read the data given the type: "Training", "Testing" """
    if dtype == "training":
        inFile = open('train-images.idx3-ubyte', 'rb')
        magic, num, rows, cols = struct.unpack(">IIII", inFile.read(16))
        temp = np.fromfile(inFile, dtype=np.uint8).reshape(num, rows, cols)
        inFile.close()
        data = []
        for img in temp:
            data.append(img.reshape((784, 1)))
            data[-1] = data[-1]/255.0
        data = np.array(data)
        inFile = open('train-labels.idx1-ubyte', 'rb')
        magic, num = struct.unpack(">II", inFile.read(8))
        labels = np.fromfile(inFile, dtype=np.uint8)
        inFile.close()
    elif dtype == "testing":
        inFile = open('t10k-images.idx3-ubyte', 'rb')
        magic, num, rows, cols = struct.unpack(">IIII", inFile.read(16))
        temp = np.fromfile(inFile, dtype=np.uint8).reshape(num, rows, cols)
        inFile.close()
        data = []
        for img in temp:
            data.append(img.reshape((784, 1)))
            data[-1] = data[-1]/255.0
        data = np.array(data)
        inFile = open('t10k-labels.idx1-ubyte', 'rb')
        magic, num = struct.unpack(">II", inFile.read(8))
        labels = np.fromfile(inFile, dtype=np.uint8)
        inFile.close()
    else:
        raise ValueError("Well, this is akward ! Please specify either training\
                         or testing data!\n")
    return (data, labels)


training_input, temp_training_labels = read("training")
testing_input, temp_testing_labels = read("testing")
# Converting to list for json writing
training_input = training_input.tolist()
training_labels = np.zeros((len(temp_training_labels), 10, 1))
for tl, num in zip(training_labels, temp_training_labels):
    tl[num][0] = 1
training_labels = training_labels.tolist()
testing_input = testing_input.tolist()
testing_labels = np.zeros((len(temp_testing_labels), 10, 1))
for tl, num in zip(testing_labels, temp_testing_labels):
    tl[num][0] = 1
testing_labels = testing_labels.tolist()

# Zipping the inputs and expected outputs together
combined_input = training_input + testing_input
combined_labels = training_labels + testing_labels
combined_data = zip(combined_input, combined_labels)
random.shuffle(combined_data)
training_data = combined_data[:60000]
testing_data = combined_data[-10000:]
data = {"training": training_data}
# Storing in json
f = open("trainingData.json", 'w')
json.dump(data, f)
f.close()
data = {"testing": testing_data}
f = open("testingData.json", 'w')
json.dump(data, f)
f.close()
