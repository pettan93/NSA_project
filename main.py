import numpy as np
from PIL import Image
import os
import random


def remove_borders(matrix, tol = 200):
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
    img = Image.open(path).convert('LA')
    data = [x[0] for x in img.getdata()]
    data = remove_borders(np.matrix(data).reshape((32, 32)))
    data_img = Image.fromarray(data)
    data_img = data_img.resize((32, 32)).convert("LA")
    data = [(255 - x[0]) / 255 for x in data_img.getdata()]
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
    return (input_matrix, np.array([one_hot(x['output'], len(labels)) for x in data]))


def split(input_data, train, test):
    training_set = input_data[:int(len(input_data) * train)]
    rest = input_data[int(len(input_data) * train):]
    test_set = rest[:int(len(rest) * test)]
    validation_set = rest[int(len(rest) * test):]
    return training_set, validation_set, test_set


def bias_variance_plot(max_neurons, input_size, output_size, train, validation, test):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    import matplotlib.pyplot as plt
    error_history = []
    plt.ion()
    plt.ylabel("Chyba")
    plt.xlabel("Počet neuronů")
    for number_of_neurons in range(50, max_neurons + 1):
        with MultilayerPerceptron(input_size, number_of_neurons, output_size) as neural_net:
            neural_net.train(train, 1e-2, validation, 10000)
            error_history.append(100 - neural_net.accuracy(test))
            plt.plot(error_history, 'r-')
            plt.pause(0.00001)

if __name__ == '__main__':
    input_data = []
    labels = []
    class_number = 0
    print("Načítám data")
    for size in ["lowercase", "uppercase"]:
        for folder in os.listdir("./resources/output/alphabet_3/%s" % size):
            labels.append(folder)
            for sample in os.listdir("./resources/output/alphabet_3/%s/%s" % (size, folder)):
                input_data.append({
                    "input": image_to_vector("./resources/output/alphabet_3/%s/%s/%s" % (size, folder, sample)),
                    "output": class_number,
                    "filename": sample
                })
            class_number += 1

    print("Načetl jsem %s obrázků" % (len(input_data)))
    print("Počet labelů %s" % (len(labels)))
    print(labels)
    random.shuffle(input_data)
    train, validation, test = split(input_data, 60 / 100, 50 / 100)
    train = prepare_for_neural_network(train)
    validation = prepare_for_neural_network(validation)
    test = prepare_for_neural_network(test)
    bias_variance_plot(200, 32 * 32, len(labels), train, validation, test)
    """
    with MultilayerPerceptron(32 * 32, 50, len(labels)) as neural_net:
        neural_net.train(train, 1e-2, validation, 100000)
        # neural_net.load("./2017-04_20_23_09/model.ckpt")

        print("Přesnost na testovacích datech %s" % neural_net.accuracy(test))
        print("Přesnost na trénovacích datech %s" % neural_net.accuracy(train))

        for i in range(10):
            indx = random.randint(0, len(input_data) - 1)
            j, k = (neural_net.feed_forward(np.matrix(input_data[indx]["input"]))[0], input_data[indx]["output"])
            print("Neuronka si myslí, že vzorek je %s skutečnost je %s" % (labels[j], labels[k]))
    """
