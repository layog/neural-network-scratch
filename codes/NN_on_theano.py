#!/usr/bin/env python
import numpy as np
import theano
import theano.tensor as T
from theano import function
import json
import time
import matplotlib.pyplot as plt

time_start = time.time()
def readData(filename):
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    return data["training"]


def checkData(images, labels):
    fig = plt.figure()
    image, label = images[0], labels[0]
    ax = fig.gca()
    fig.show()
    try:
        for image, label in zip(images, labels):
            image = np.array(image)
            # print np.shape(image)
            image = image.reshape((28, 28))
            # print image[0:100]
            image = image*255.0
            image.astype(int)
            ax.imshow(image, cmap="gray")
            print np.argmax(label)
            fig.canvas.draw()
            raw_input()
    except KeyboardInterrupt:
        plt.close(fig)
        raise


class Network(object):
    def __init__(self, sizes, shared_inputs=True):
        self.shared_inputs = shared_inputs
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.X = T.fmatrix('inputs')
        self.y = T.fmatrix('onehot_outputs')
        self.train_size = np.float32(0)
        self.valid_size = np.float32(0)
        self.test_size = np.float32(0)
        self.batch_size = np.int32(0)
        self.num_batches = np.float32(0)
        self.initialized = False

    def weight_init(self):
        self.check_biases = [np.random.randn(y).astype('float32')
                             for y in self.sizes[1:]]
        self.check_weights = [np.random.randn(x, y).astype('float32')
                              for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [theano.shared(y.astype('float32'))
                       for y in self.check_biases]
        self.weights = [theano.shared(x.astype('float32'))
                        for x in self.check_weights]

    def expressions(self, index, batch_size, alpha, lmbda):
        self.weight_init()
        self.train_size = shared_train_set_x.get_value(borrow=True).shape[0]
#         self.valid_size = shared_valid_set_x.get_value(borrow=True).shape[0]
#         self.test_size = shared_test_set_x.get_value(borrow=True).shape[0]
        self.batch_size = batch_size
        self.num_batches = self.train_size/self.batch_size
        self.epsilon = theano.shared(alpha)  # Set learning rate
        self.reg_lambda = theano.shared(lmbda)  # Set regularization constant
        self.weighted_sum = []
        self.layers_outputs = []
        sum_of_weights = theano.shared(np.float32(0.0))
        for i in range(len(self.sizes) - 1):
            sum_of_weights += T.sum(T.sqr(self.weights[i]))
            if(i == 0):
                self.weighted_sum.append(T.dot(self.X, self.weights[i]) +
                                         self.biases[i])
            else:
                self.weighted_sum.append(T.dot(self.layers_outputs[-1],
                                               self.weights[i]) +
                                         self.biases[i])
#             if(i != len(self.sizes) - 2):
            if(True):
                self.layers_outputs.append(T.nnet.sigmoid(
                    self.weighted_sum[-1]))
#             else:
#                 self.layers_outputs.append(T.nnet.softmax(self.weighted_sum[-1]))
        # Implement the sum for loss_reg
        answers = T.argmax(self.layers_outputs[-1], axis=1)
        self.get_output = function([], answers, givens={
            self.X: shared_train_set_x
        })
        expected = T.argmax(self.y, axis=1)
        self.get_expected = function([], expected, givens={
            self.y: shared_train_set_y
        })
        # self.get_output = function([], self.layers_outputs[-1], givens={
        #     self.X: shared_train_set_x
        # })
        loss_reg = np.float32(1.)/self.train_size *\
            self.reg_lambda/np.float32(2) * sum_of_weights
        # loss = T.nnet.categorical_crossentropy(self.layers_outputs[-1],
        #                                        self.y).mean() + loss_reg
        loss = T.mean((-(self.y*T.log(self.layers_outputs[-1]) + (1 - self.y) *
                         T.log(1 - self.layers_outputs[-1])))) + loss_reg
        # self.calculate_loss = function([], loss, givens={
        #     self.X: shared_train_set_x,
        #     self.y: shared_train_set_y
        # })
        self.calculate_loss = function([], loss-loss_reg, givens={
            self.X: shared_train_set_x,
            self.y: shared_train_set_y
        })
        self.gradWeights = []
        self.gradBiases = []
        updates = []
        for w, b in zip(self.weights, self.biases):
            self.gradWeights.append(T.grad(loss, w))
            self.gradBiases.append(T.grad(loss, b))
            updates.append((w, w - self.epsilon*self.gradWeights[-1]))
            updates.append((b, b - self.epsilon*self.gradBiases[-1]))
        self.gradient_step = function([index], updates=updates, givens={
            self.X: shared_train_set_x[index*self.batch_size: (index+1)*self.batch_size],
            self.y: shared_train_set_y[index*self.batch_size: (index+1)*self.batch_size]
        })
        self.initialized = True

    def train(self, epochs, alphas, breaks):
        print "The accuracy before training : " +\
            str(sum(map(float, self.get_output() == self.get_expected()))*100 /
                len(y))
        i = 0
        breaksDone = True
        if(not alphas):
            breaksDone = False
        for epoch in xrange(epochs):
            if(not breaksDone):
                if(epoch == breaks[i]):
                    self.epsilon = theano.shared(alphas[i])
                    i += 1
                if(i == len(alphas)):
                    breaksDone = True
            time1 = time.time()
            for j in xrange(self.num_batches):
                self.gradient_step(j)
            # if(epoch % 1000 == 0):
            print "The cost after epoch {} is : ".format(epoch+1)\
                + str(self.calculate_loss())
            timeTook.append(time.time() - time1)
            print "Time Took: ", timeTook[-1]
        print "The accuracy after training : " +\
            str(sum(map(float, self.get_output() == self.get_expected()))*100 /
                len(y))
        f = open("timeTookGPU.json", "w")
        json.dump({"timeTookGPU": timeTook}, f)
        f.close()
        print "Training Complete"


def train_network(net, alpha, lmbda, batch_size, alphas, breaks, epochs):
    # Implement the accuracy function
    net.expressions(T.iscalar('index'), np.int32(batch_size),
                    np.float32(alpha), np.float32(lmbda))
    net.train(epochs, alphas, breaks)
    # print accuracy after the training


data = readData("trainingData.json")
print "Data Reading Completes"
training_data = data[:45000]
x, y = zip(*training_data)
x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape[0], x.shape[1])
y = y.reshape(y.shape[0], y.shape[1])
# checkData(x, y)
shared_train_set_x = theano.shared(np.asarray(x).astype('float32'), name='X')
shared_train_set_y = theano.shared(np.asarray(y).astype('float32'), name='y')
print "Data copying completes"
timeTook = []
timeGPU = {}
for layers in [
    [784, 300, 150, 70, 10],
    [784, 600, 300, 150, 70, 10],
    [784, 1000, 600, 300, 150, 70, 10]
]:
    timeTook = []
    net = Network(layers)
    train_network(net, 0.5, 3.75, 15, [0.2, 0.1, 0.05, 0.025, 0.01],
                  [15, 25, 35, 40, 45], 1)
    timeGPU[len(layers)] = timeTook
f = open("no_layers_GPU.json", "w")
json.dump(timeGPU, f)
f.close()
timeGPU = {}
for batch_size in [15, 25, 35, 45, 55]:
    timeTook = []
    net = Network([784, 300, 150, 70, 10])
    train_network(net, 0.5, 3.75, batch_size, [0.2, 0.1, 0.05, 0.025, 0.01],
                  [15, 25, 35, 40, 45], 1)
    timeGPU[batch_size] = timeTook
f = open("batch_size_GPU.json", "w")
json.dump(timeGPU, f)
f.close()
# valid_data = data[45000:]
# x, y = zip(*valid_data)
# x = np.array(x)
# y = np.array(y)
# x = x.reshape(x.shape[0], x.shape[1])
# y = y.reshape(y.shape[0], y.shape[1])
# shared_train_set_x = theano.shared(np.asarray(x).astype('float32'), name='X')
# shared_train_set_y = theano.shared(np.asarray(y).astype('float32'), name='y')
# print "The accuracy on validation data is : " +\
#    str(sum(map(float, net.get_output() == net.get_expected()))*100 /
#        len(y))
print "Total Time Taken: ", time.time() - time_start
