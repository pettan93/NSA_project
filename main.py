from pprint import pprint

import numpy as np
from PIL import Image
import os
import random


def remove_borders(matrix, tol=200):
    mask = matrix < tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return matrix[x0:x1, y0:y1]


def image_to_vector(path):
    """
    Převede obrázek na vektor o velikosti vyska*sirka.
    :param path: cesta k obrázku
    :return: 
    """
    img = Image.open(path).convert('L')
    data = np.array(img)
    data = remove_borders(data)
    data_img = Image.fromarray(data)
    data_img = data_img.resize((32, 32))
    data = [(255 - x) / 255 for x in data_img.getdata()]
    return data


def one_hot(x, max_size):
    """
    Na základě třídy (celé čislo) vygeneruje tzv. one hot vektor, tedy vektor, který obsahuje jedničku na pozici třídy
    a zbytek je vyplněn 0.
    >>> one_hot(1, 2)
    [0, 1]
    >>> one_hot(0, 2)
    [1, 0]
    >>> one_hot(0, 4)
    [1, 0, 0, 0]
    :param x: klasifikovaná třída
    :param max_size: velikost vektoru
    """
    return [int(i == x) for i in range(max_size)]


def prepare_for_neural_network(data):
    """
    Připraví data pro neuronovou síť
    :param data: Data pro neuronovou síť
    :return: 
    """
    input_matrix = np.array([x['input'] for x in data], dtype=np.float32)
    return input_matrix, np.array([one_hot(x['output'], len(labels)) for x in data])


def split(input_data, train, test):
    training_set = input_data[:int(len(input_data) * train)]
    rest = input_data[int(len(input_data) * train):]
    test_set = rest[:int(len(rest) * test)]
    validation_set = rest[int(len(rest) * test):]
    return training_set, validation_set, test_set


def lambda_plot(input_size, hidden_layer_size, output_size, train, validation, test, training_length, softness = 0.01):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    import matplotlib.pyplot as plt
    plt.ion()
    plt.ylabel("J($\\theta$)")
    plt.xlabel("$\\lambda$")
    j_validation = []
    x_axis = []
    for real_lambda in reversed(np.arange(softness, 1, softness)):
        with MultilayerPerceptron(input_size, hidden_layer_size, output_size) as neural_net:
            neural_net.train(train, real_lambda, validation, training_length)
            j_validation.append(neural_net.j(test[0], test[1]))
            x_axis.append(real_lambda)
            plt.plot(x_axis, j_validation, 'b-')
            plt.gca().invert_xaxis()
            plt.pause(0.00001)
    plt.ioff()
    input('Done')


def bias_variance_plot(input_size, output_size, train, validation, test):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    from ML import Batcher
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    j_train_label = mpatches.Patch(color='red', label='$j_{trenovaci}$')
    j_validation_label = mpatches.Patch(color='blue', label='$j_{validace}$')
    plt.ion()
    plt.ylabel("J($\\theta$)")
    plt.xlabel("Pocet vzorku")
    plt.legend(handles=[j_train_label, j_validation_label])

    j_train = []
    j_validation = []
    train_set = train
    batcher = Batcher.Batcher(train[0], train[1])
    for training_percentage in range(10, 101, 5):
        with MultilayerPerceptron(input_size, 200, output_size) as neural_net:
            train = batcher.next_batch(int(len(train_set[0]) * (training_percentage / 100)))
            neural_net.train(train, 0.01, validation, 1000)
            j_train.append(neural_net.j(test[0], test[1]))
            j_validation.append(neural_net.j(validation[0], validation[1]))
            x_axis = list(range(len(j_train)))
            plt.plot(x_axis, j_train, 'r-', x_axis, j_validation, 'b-')
            plt.pause(0.00001)
    input("Konec analýzy pro pokračování stiskněte jakoukoliv klávesu")


def dump_train(input_size, output_size, train, validation, test):
    from ML.MultilayerPerceptron import MultilayerPerceptron

    with MultilayerPerceptron(input_size, 900, output_size) as neural_net:
        neural_net.train(train, 0.001, validation, 60000)
        print("Error output ->")
        print(neural_net.error(test),"%")
    print("Neuronka naučená.")


def plot_data(train, validation, test, labels):
    # what the hell, piece of the most ineffective code in galaxy starts here..
    d_train = dict()
    d_validation = dict()
    d_test = dict()
    for sample in train:
        if sample["class"] not in d_train:
            d_train[sample["class"]] = 0
        d_train[sample["class"]] += 1
    for sample in validation:
        if sample["class"] not in d_validation:
            d_validation[sample["class"]] = 0
        d_validation[sample["class"]] += 1
    for sample in test:
        if sample["class"] not in d_test:
            d_test[sample["class"]] = 0
        d_test[sample["class"]] += 1

    a_train = list()
    for l in labels:
        a_train.append(d_train[l])

    a_val = list()
    for l in labels:
        a_val.append(d_validation[l])

    a_test = list()
    for l in labels:
        a_test.append(d_test[l])
    # survived? gz

    import numpy as np
    import matplotlib.pyplot as plt

    N = len(labels)

    ind = np.arange(N)
    width = 0.28

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, a_train, width, color='r')
    rects2 = ax.bar(ind + width, a_val, width, color='y')
    rects3 = ax.bar(ind + (width * 2), a_test, width, color='b')

    ax.set_ylabel('Number of samples')
    ax.set_title('Data splited to  train, cross-validation and test')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)

    ax.legend((rects1[0], rects2[0], rects3[0]), ('Train', 'Validation', "Test"))

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.show()


if __name__ == '__main__':
    input_data = []
    labels = []
    class_number = 0
    samples_limit = 1000
    print("Načítám data")
    for size in ["lowercase"]:
        for folder in os.listdir("./resources/output/alphabet_1/%s" % size):
            labels.append(folder)
            for i, sample in enumerate(os.listdir("./resources/output/alphabet_1/%s/%s" % (size, folder))):
                if i is samples_limit:
                    break
                input_data.append({
                    "input": image_to_vector("./resources/output/alphabet_1/%s/%s/%s" % (size, folder, sample)),
                    "output": class_number,
                    "class": folder
                })
            class_number += 1
    print("Načetl jsem %s obrázků" % (len(input_data)))
    print("Počet labelů %s" % (len(labels)))
    print(labels)
    random.shuffle(input_data)

    train, validation, test = split(input_data, 60 / 100, 50 / 100)

    # plot_data(train, validation, test, labels)

    train = prepare_for_neural_network(train)
    validation = prepare_for_neural_network(validation)
    test = prepare_for_neural_network(test)

    # bias_variance_plot(32 * 32, len(labels), train, validation, test)


    #dump_train(32 * 32, len(labels), train, validation, test)
    lambda_plot(32 * 32, 300, len(labels), train, validation, test, 10000)


