import numpy as np
from PIL import Image
import os
import random


def image_to_vector(path):
    """
    Převede obrázek na vektor o velikosti vyska*sirka.
    :param path: cesta k obrázku
    :return: 
    """
    img = Image.open(path).convert('LA')
    data = [x[0] for x in img.getdata()]
    maximum = max(data) 
    data = [(255-x) / maximum for x in data]
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
    input_matrix = np.array([x['input'] for x in data])
    return (input_matrix, np.array([one_hot(x['output'], len(labels)) for x in data]))

if __name__ == '__main__':
    input_data = []
    labels = []
    class_number = 0
    print("Načítám data")
    for size in ["lowercase", "uppercase"]:
        for folder in os.listdir("./resources/output/alphabet/%s" % size):
            labels.append(folder)
            for sample in os.listdir("./resources/output/alphabet/%s/%s" % (size, folder)):
                input_data.append({
                    "input": image_to_vector("./resources/output/alphabet/%s/%s/%s" % (size, folder, sample)),
                    "output": class_number,
                    "filename": sample
                })
            class_number += 1

    for size in ["lowercase", "uppercase"]:
        for folder in os.listdir("./resources/output/alphabet_2/%s" % size):
            labels.append(folder)
            for sample in os.listdir("./resources/output/alphabet_2/%s/%s" % (size, folder)):
                input_data.append({
                    "input": image_to_vector("./resources/output/alphabet_2/%s/%s/%s" % (size, folder, sample)),
                    "output": class_number,
                    "filename": sample
                })
            class_number += 1

    print("Načetl jsem %s obrázků" % (len(input_data)))
    print("Počet labelů %s" % (len(labels)))
    print(labels)
    random.shuffle(input_data)
    training_set = input_data[:int(len(input_data) * 0.9)]
    training_set = prepare_for_neural_network(training_set)

    classification_data = input_data[int(len(input_data) * 0.1):]
    classification_data = prepare_for_neural_network(classification_data)

    from ML.MultilayerPerceptron import MultilayerPerceptron

    with MultilayerPerceptron(100 * 100, 400, len(labels)) as neural_net:
        neural_net.train(training_set, 1e-2, classification_data, 300)
        #neural_net.load("./2017-04_21_07_55/model.ckpt")

        print("Přesnost na klasifikačních datech %s" % neural_net.accuracy(classification_data))
        print("Přesnost na trénovacích datech %s" % neural_net.accuracy(training_set))

        for i in range(10):
            indx = random.randint(0, len(input_data) - 1)
            j, k = (neural_net.feed_forward(np.matrix(input_data[indx]["input"]))[0], input_data[indx]["output"])
            print("Neuronka si myslí, že vzorek je %s skutečnost je %s" % (labels[j], labels[k]))
