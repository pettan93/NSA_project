import numpy as np
import tensorflow as tf
from scipy.optimize import *
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


class Batcher:
    def __init__(self, arr):
        self.array = arr

    def next_batch(self, size):
        random.shuffle(self.array)
        return self.array[:size]


class NeuralNetwork:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, input_size, output_size):
        self.hidden_layer = tf.Variable(tf.zeros([input_size, output_size]))
        self.b = tf.Variable(tf.zeros([output_size]))
        self.input_size = input_size
        self.output_size = output_size
        self.session = tf.InteractiveSession()

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        x = tf.placeholder(tf.float32, [None, self.input_size])
        feed_forward = tf.argmax(tf.nn.softmax(tf.matmul(x, self.hidden_layer) + self.b), 1)
        return self.session.run(feed_forward, feed_dict={x: input_data})

    def train(self, training_set, learning_rate, epochs):
        x = tf.placeholder(tf.float32, [None, self.input_size])
        y_ = tf.placeholder(tf.float32, [None, self.output_size])
        feed_forward = tf.nn.softmax(tf.matmul(x, self.hidden_layer) + self.b)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(feed_forward), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        tf.global_variables_initializer().run()
        for i in range(epochs):
            self.session.run(train_step, feed_dict={x: training_set[0], y_: training_set[1]})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Končím")
        self.session.close()
        return False


def image_to_vector(path):
    img = Image.open(path).convert('LA')
    data = [x[0] for x in img.getdata()]
    maximum = max(data) 
    data = [(255 - x) / maximum for x in data]
    return data


def one_hot(x, max_size):
    return [int(i == x) for i in range(max_size)]

if __name__ == '__main__':
    input_data = []
    labels = []
    class_number = 0
    for folder in os.listdir("./resources/output/alphabet/lowercase"):
        labels.append(folder)
        for sample in os.listdir("./resources/output/alphabet/lowercase/%s" % folder):
            input_data.append({
                "input": image_to_vector("./resources/output/alphabet/lowercase/%s/%s" % (folder, sample)),
                "output": class_number,
                "filename": sample
            })
        class_number += 1

    for folder in os.listdir("./resources/output/alphabet/uppercase"):
        labels.append(folder)
        for sample in os.listdir("./resources/output/alphabet/uppercase/%s" % folder):
            input_data.append({
                "input": image_to_vector("./resources/output/alphabet/uppercase/%s/%s" % (folder, sample)),
                "output": class_number,
                "filename": sample
            })
        class_number += 1

    neural_net = NeuralNetwork(100 * 100, len(labels))
    random.shuffle(input_data)
    training_set = input_data[:int(len(input_data) * 0.8)]
    input_matrix = np.array([x['input'] for x in training_set])

    training_set = ((input_matrix), np.array([one_hot(x['output'], len(labels)) for x in training_set]))

    neural_net.train(training_set, 0.5, 100)

    classified = 0
    correctly_classified = 0
    classification_data = input_data[int(len(input_data) * 0.1):]
    for sample in classification_data:
        classified += 1
        guess = neural_net.feed_forward(np.matrix(sample["input"]))[0]
        if guess == sample["output"]:
            correctly_classified += 1

    print("Úspěšnost %s " % str((correctly_classified / classified) * 100))

    for i in range(10):
        indx = random.randint(0, len(input_data) - 1)
        j, k = (neural_net.feed_forward(np.matrix(input_data[indx]["input"]))[0], input_data[indx]["output"])
        print("Neuronka si myslí, že vzorek je %s skutečnost je %s" % (labels[j], labels[k]))

    fig = plt.imshow(np.matrix(input_data[indx]["input"]).reshape((100, 100)))
    plt.show()

