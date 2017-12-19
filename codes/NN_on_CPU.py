#!/usr/bin/env python
"""A basic implementation of Neural Network on CPU"""

import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt


costsPerEpoch = []  # Will hold the costs used later for plotting


def readTestData():
    inp = [[[0], [0], [0]], [[0], [0], [1]], [[0], [1], [0]], [[0], [1], [1]],
           [[1], [0], [0]], [[1], [0], [1]], [[1], [1], [0]], [[1], [1], [1]]]
    ORout = [[[0]]] + [[[1]]]*7
    # ANDout = [[[0]]]*7 + [[[1]]]
    # XORout = [[[0]], [[1]], [[1]], [[0]], [[1]], [[0]], [[0]], [[1]]]
    return (zip(inp, ORout), zip(inp, ORout))


def readData(filename):
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    return data["training"]


def sigmoid(z):
    """ The sigmoid function """
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    """ Derivative of the Sigmoid funciton """
    return sigmoid(z) * (1.0 - sigmoid(z))


class QuadraticCost(object):
    """Class for defining quadratic cost and its delta"""
    @staticmethod
    def fn(a, y):
        """ Return the cost given the expected output """
        return 0.5*np.linalg.norm(a-y)**2
        # or we can write 0.5*np.sum((a-y)**2)

    @staticmethod
    def delta(z, a, y):
        """ Return the derivative of the quadratic cost function """
        return (a-y)*sigmoid_prime(z)


class CrossEntropyCost(object):
    """Class for defining cost entropy cost and its delta"""
    @staticmethod
    def fn(a, y):
        """ Return the cost given the expected output """
        a = np.array(a)
        y = np.array(y)
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """ Return the derivative of the cross entropy cost function """
        return (a-y)


def checkData(data):
    fig = plt.figure()
    image, label = data[0]
    ax = fig.gca()
    fig.show()
    try:
        for image, label in data:
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
    """Main class to create the Neural Network"""
    def __init__(self, sizes, cost=CrossEntropyCost, load=False, monitor=True):
        """ The sizes contains the size of each neural layer and the cost
            specifies the type of cost to be associated with a particular
            network: Quadratic Cost or Cross Entropy Cost. The biases and
            weights are initialized using another function. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializer(load)
        self.cost = cost
        self.monitor = monitor

    def weight_initializer(self, load):
        """ Initialize each weight using a Gaussian distribution with mean 0
            and standard deviation 1/sqrt(no. of input neurons) """
        if not load:
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in
                            zip(self.sizes[:-1], self.sizes[1:])]
            return
        f = open("theta.json", "r")
        theta = json.load(f)
        self.biases = theta["biases"]
        self.weights = theta["weights"]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data,  mini_batch_size, epochs, alpha,
            alphas, breaks, lmbda):
        try:
            assert(len(alphas) == len(breaks))
        except AssertionError:
            print "This is akward!"
            raise
        if(len(alphas) != 0):
            alpha_no = 0
            remain = True
        else:
            remain = False
        print "Starting the network training"
        n = len(training_data)
        for epoch in xrange(epochs):
            if(remain):
                if(epoch == breaks[alpha_no]):
                    alpha = alphas[alpha_no]
                    alpha_no += 1
                    if (alpha_no == len(alphas)):
                        remain = False
            print "Training for epoch {} starts".format(epoch+1)
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            time1 = time.time()
            for mini_batch in mini_batches:
                self.update_mini_batch2(mini_batch, alpha, lmbda, n)
            time2 = time.time()
            print "Time took to train: ", (time2 - time1)
            timeTook.append(time2 - time1)
            print "Epoch {} training complete".format(epoch+1)
            if(self.monitor):
                costsPerEpoch.append(self.netcost(training_data))
                print "The cost after this epoch is: ",
                print costsPerEpoch[-1]
        print "Training Complete !"
        f = open("timeTookMatrix.json", "w")
        json.dump({"timeTookMatrix": timeTook}, f)
        f.close()

    def update_mini_batch(self, mini_batch, alpha, lmbda, n):
        """Linear implementation of mini batch update

        Every example in a mini batch is backproped individually
        """
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            x = np.array(x)
            y = np.array(y)
            delta_change_b, delta_change_w = self.backprop(x, y)
            change_b = [cb+dcb for cb, dcb in zip(change_b, delta_change_b)]
            change_w = [cw+dcw for cw, dcw in zip(change_w, delta_change_w)]
        self.weights = [((1-alpha*(lmbda/n))*w - (alpha/(len(mini_batch)))*cw)
                        for w, cw in zip(self.weights, change_w)]
        self.biases = [b - (alpha/(len(mini_batch)))*cb
                       for b, cb in zip(self.biases, change_b)]

    def update_mini_batch2(self, mini_batch, alpha, lmbda, n):
        """Matrix implementation of mini batch update

        A complete mini batch is backproped in one pass
        """
        x, y = zip(*mini_batch)
        x = np.array(x)
        y = np.array(y)
        combined_x = x.transpose()[0]
        combined_y = y.transpose()[0]
        change_b, change_w = self.backprop(combined_x, combined_y)
        self.weights = [((1-alpha*(lmbda/n))*w - (alpha/(len(mini_batch)))*cw)
                        for w, cw in zip(self.weights, change_w)]
        self.biases = [b - (alpha/(len(mini_batch)))*cb
                       for b, cb in zip(self.biases, change_b)]

    def numericalGradient(self, x, y):
        """To test the implementation of backprop is correct
        """
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]
        epsilon = 1e-4
        for i, bias in enumerate(self.biases):
            for j, b in enumerate(bias):
                for k, t in enumerate(b):
                    temp = t-epsilon
                    self.biases[i][j][k] = temp
                    c1 = (self.cost).fn(self.feedforward(x), y)
                    temp = t+epsilon
                    self.biases[i][j][k] = temp
                    c2 = (self.cost).fn(self.feedforward(x), y)
                    self.biases[i][j][k] = t
                    change_b[i][j][k] = (c2-c1)/(2*epsilon)
        for i, weight in enumerate(self.weights):
            for j, w in enumerate(weight):
                for k, t in enumerate(w):
                    temp = t-epsilon
                    self.weights[i][j][k] = temp
                    c1 = (self.cost).fn(self.feedforward(x), y)
                    temp = t+epsilon
                    self.weights[i][j][k] = temp
                    c2 = (self.cost).fn(self.feedforward(x), y)
                    self.weights[i][j][k] = t
                    change_w[i][j][k] = (c2-c1)/(2*epsilon)
        return (change_b, change_w)

    def backprop(self, x, y):
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass - feedforward
        activation = x
        activations = [x]  # List to store all the activations
        zs = []  # List to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass - backprop
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        change_b[-1] = delta
        change_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot((self.weights[-l+1]).transpose(), delta)*sp
            change_b[-l] = delta
            # The following two lines are required for update_mini_batch2
            # Comment these out if using update_mini_batch
            temp = np.sum(change_b[-l+1], 1)
            change_b[-l+1] = temp.reshape((len(temp), 1))
            #
            change_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # The following two lines are required for update_mini_batch2
        temp = np.sum(change_b[0], 1)
        change_b[0] = temp.reshape((len(temp), 1))
        #
        return (change_b, change_w)

    def netcost(self, tdata):
        x, y = zip(*tdata)
        x = np.array(x)
        y = np.array(y)
        combined_x = x.transpose()[0]
        combined_y = y.transpose()[0]
        output = self.feedforward(combined_x)
        cost = (self.cost).fn(output, combined_y)
        return cost

    def accuracy(self, tdata):
        x, y = zip(*tdata)
        x = np.array(x)
        y = np.array(y)
        combined_x = x.transpose()[0]
        combined_y = y.transpose()[0]
        output = self.feedforward(combined_x)
        output = np.argmax(output, 0)
        y = np.argmax(combined_y, 0)
        total = len(output)
        ans = np.sum(output == y)
        return ans*100.0/total

    def getOutput(self, data):
        for inp, label in data:
            print "Expected", label
            print "Output", self.feedforward(inp)
            print "-----------------------------"

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": (self.cost).__name__}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def train_network(ftry, training_data, mini_batch, epochs, alphas, breaks,
                  lmbda):
    ftry.SGD(training_data, mini_batch, epochs, alphas[0], alphas[1:], breaks,
             lmbda)
    return ftry


if __name__ == '__main__':
    lmbda = 3.75  # 7.4976/2 turns out to be the optimum
    mini_batch_size = 15
    epochs = 1

    # Different alphas are defined to use after certain batch size, indicated by breaks
    # A function to update alpha (learning rate) can also be used
    alphas = [0.5, 0.2, 0.1, 0.05, 0.025, 0.01]
    breaks = [15, 25, 35, 40, 45]

    data = readData("trainingData.json")
    print "Data Reading Completes"
    opti_data = data[45000:]
    training_data = data[:45000]
    timeMatrix = {}
    for layers in [
        [784, 300, 150, 70, 10],
        [784, 600, 300, 150, 70, 10],
        [784, 1000, 600, 300, 150, 70, 10]
    ]:
        timeTook = []
        ntry = Network(layers, monitor=False)
        time1 = time.time()
        ntry = train_network(ntry, training_data, mini_batch_size, epochs,
                             alphas, breaks, lmbda)
        time2 = time.time()
        timeMatrix[len(layers)] = timeTook
        print "Total training time:", (time2 - time1)
    f = open('no_layers_matrix.json', 'w')
    json.dump(timeMatrix, f)
    f.close()
    # print "Accuracy on training_data", ntry.accuracy(training_data)
    # print "Accuracy on Cross Validation data", ntry.accuracy(opti_data)
