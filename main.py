import numpy as np
import tensorflow as tf
from scipy.optimize import *
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    return np.multiply(x, (1.0 - x))


class NeuralNetwork:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, input_size, output_size):
        self.hidden_layer = tf.Variable(tf.zeros([input_size, output_size]))
        self.b = b = tf.Variable(tf.zeros([output_size]))
        self.input_size = input_size
        self.output_size = output_size
        self.session = tf.InteractiveSession()

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        result = None
        x = tf.placeholder(tf.float32, [None, self.input_size])
        feed_forward = tf.argmax(tf.nn.softmax(tf.matmul(x, self.hidden_layer) + self.b))
        tf.global_variables_initializer().run()
        with tf.Session() as sess:
            return sess.run(result, feed_dict = {x: input_data})


    def train(self, training_set, learning_rate, epochs):
        x = tf.placeholder(tf.float32, [None, self.input_size])
        y_ = tf.placeholder(tf.float32, [None, self.output_size])
        feed_forward = tf.nn.softmax(tf.matmul(x, self.hidden_layer) + self.b)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(feed_forward), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        tf.global_variables_initializer().run()
        for i in range(epochs):
            self.session.run(train_step, feed_dict = {x: training_set[0], y_: training_set[1]})



def image_to_vector(path):
    img = Image.open(path).convert('LA')
    data = [x[0] for x in img.getdata()]
    maximum = max(data) 
    data = [x / maximum for x in data]
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
        if class_number >= 2:
            break
    

    neural_net = NeuralNetwork(100 * 100, len(labels))

    random.shuffle(input_data)
    training_set = input_data[:int(len(input_data) * 0.8)]
    input_matrix = np.array([x['input'] for x in training_set])
    
    training_set = ((input_matrix), np.array([one_hot(x['output'], len(labels)) for x in training_set]))
    
    neural_net.train(training_set, 0.5, 400)

    classified = 0
    correctly_classified = 0
    classification_data = input_data[int(len(input_data) * 0.8):]
    for sample in classification_data:
        classified += 1
        guess = neural_net.feed_forward(np.matrix(sample["input"]))
        print(guess)
        if guess == sample["output"]:
            correctly_classified += 1


    print("Úspěšnost %s " % str((correctly_classified / classified) * 100))

    for i in range(5):
        indx = random.randint(0, len(input_data) - 1)
        fig = plt.imshow(input_data[indx]["input"].reshape((100, 100)))
        plt.show()
        j, k = (neural_net.feed_forward(np.matrix(input_data[indx]["input"])), input_data[indx]["output"])
        print("Neuronka si myslí, že vzorek je %s skutečnost je %s" % (labels[j], labels[k]))
