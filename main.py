import os
import random

import numpy as np
from PIL import Image


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


def lambda_plot(input_size, hidden_layer_size, output_size, train, validation, test, epochs, softness=0.01, step=0.01, stop=1):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    import matplotlib.pyplot as plt
    plt.ion()
    plt.ylabel("Přesnost [%]")
    plt.xlabel("$\\lambda$")
    accuracy = []
    x_axis = []
    for real_lambda in reversed(np.arange(softness, stop, step
                                          )):
        with MultilayerPerceptron(input_size, hidden_layer_size, output_size) as neural_net:
            neural_net.train(train, real_lambda, validation, epochs)
            accuracy.append(neural_net.accuracy(test))
            x_axis.append(real_lambda)
            plt.plot(x_axis, accuracy, 'b-')
            # plt.gca().invert_xaxis()
            plt.pause(0.00001)
    plt.ioff()
    input('Done')


def epoch_plot(input_size, hidden_layer_size, output_size, train, validation, test, real_lambda, training_length):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    import matplotlib.pyplot as plt
    plt.ion()
    plt.ylabel("Přesnost [%]")
    plt.xlabel("Počet epoch")
    accuracy = []
    x_axis = []
    last_accuracy = None
    for length in range(1, training_length + 1, 500):
        print(length)
        with MultilayerPerceptron(input_size, hidden_layer_size, output_size) as neural_net:
            neural_net.train(train, real_lambda, validation, length)
            acc = neural_net.accuracy(test)
            accuracy.append(acc)
            last_accuracy = acc
            x_axis.append(length)
            plt.plot(x_axis, accuracy, 'b-')
            plt.pause(0.00001)
    plt.ioff()
    print('Done, Last accuracy : ', last_accuracy, "%")
    input("Press Enter to exit")

def neural_plot(input_size, output_size, train, validation, epochs, learning_rate, starting_number_of_neurons, step, end_number_of_neurons):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    import matplotlib.pyplot as plt
    plt.ion()
    plt.ylabel("Přesnost [%]")
    plt.xlabel("Počet neuronů")
    accuracy = []
    x_axis = []
    last_accuracy = None
    for neurons in range(starting_number_of_neurons, end_number_of_neurons + 1, step):
        with MultilayerPerceptron(input_size, neurons, output_size) as neural_net:
            neural_net.train(train, learning_rate, validation, epochs)
            acc = neural_net.accuracy(test)
            accuracy.append(acc)
            last_accuracy = acc
            x_axis.append(neurons)
            plt.plot(x_axis, accuracy, 'b-')
            plt.pause(0.00001)
    plt.ioff()
    print('Done, Last accuracy : ', last_accuracy, "%")
    input("Press Enter to exit")


def bias_variance_plot(input_size, output_size, train, validation, test, neurons, alpha_rate, epochs):
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
        with MultilayerPerceptron(input_size, neurons, output_size) as neural_net:
            train = batcher.next_batch(int(len(train_set[0]) * (training_percentage / 100)))
            neural_net.train(train, alpha_rate, validation, epochs)
            j_train.append(neural_net.j(test[0], test[1]))
            j_validation.append(neural_net.j(validation[0], validation[1]))
            x_axis = list(range(len(j_train)))
            plt.plot(x_axis, j_train, 'r-', x_axis, j_validation, 'b-')
            plt.pause(0.00001)
    input("Konec analýzy pro pokračování stiskněte jakoukoliv klávesu")


def dump_train(input_size, output_size, train, validation, test, neurons, alpha, epochs):
    from ML.MultilayerPerceptron import MultilayerPerceptron

    neural_net = MultilayerPerceptron(input_size, neurons, output_size)
    neural_net.train(train, alpha, validation, epochs)
    print("Vypočtená chyba - neural_net.error : ", neural_net.error(test), "%")
    print("Neuronka naučená.")
    return neural_net


def naive_accuracy_test_from_drive(neural_network, alphabet_number, samples, verbose):
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u',
              'v', 'w', 'x', 'y', 'z']
    total = 0
    success = 0
    if not verbose:
        print("Testuji..")

    while total < samples:
        input_char = random.choice(labels)
        path = random.choice(
            os.listdir("./resources/output/%s/lowercase/%s" % (alphabet_number, input_char)))
        path = "./resources/output/%s/lowercase/%s/" % (alphabet_number, input_char) + path
        image_vector = image_to_vector(path)
        data = np.array(image_vector).reshape(1, 1024)
        indx = neural_network.feed_forward(data)
        if verbose:
            print(total, "[" + input_char + "] => [" + labels[indx[0]] + "]")
        total += 1
        if labels[indx[0]] is input_char:
            success += 1
        if verbose:
            print("Úspěšnost [" + str(success) + "/" + str(total) + "] - " + str((success / total) * 100) + " %")
    if not verbose:
        print("Úspěšnost [" + str(success) + "/" + str(total) + "] - " + str((success / total) * 100) + " %")


def naive_accuracy_test(neural_network, test_data, verbose):
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u',
              'v', 'w', 'x', 'y', 'z']
    total = 0
    success = 0

    if not verbose:
        print("Testuji..")

    for entry in test_data:
        indx = neural_network.feed_forward(np.array(entry["input"]).reshape(1, 1024))
        if verbose:
            print(total, "[" + entry["class"] + "] => [" + labels[indx[0]] + "]")
        total += 1
        if labels[indx[0]] is entry["class"]:
            success += 1
        if verbose:
            print("Úspěšnost [" + str(success) + "/" + str(total) + "] - " + str((success / total) * 100) + " %")

    if not verbose:
        print("Úspěšnost [" + str(success) + "/" + str(total) + "] - " + str((success / total) * 100) + " %")


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
        if l in d_train:
            a_train.append(d_train[l])
        else:
            a_train.append(0)

    a_val = list()
    for l in labels:
        if l in d_validation:
            a_val.append(d_validation[l])
        else:
            a_val.append(0)

    a_test = list()
    for l in labels:
        if l in d_test:
            a_test.append(d_test[l])
        else:
            a_test.append(0)

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


def bias_variance_plot_debug(input_size, output_size, train, validation, test, neurons, alpha_rate, epochs):
    from ML.MultilayerPerceptron import MultilayerPerceptron
    from ML import Batcher
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    j_train_label = mpatches.Patch(color='red', label='$j_{trenovaci}$')
    j_validation_label = mpatches.Patch(color='blue', label='$j_{validace}$')
    plt.ion()
    plt.ylabel("J($\\theta$)")
    plt.xlabel("Zpracováno vozorků [%] ")
    plt.legend(handles=[j_train_label, j_validation_label])

    x_axis = []

    j_train = []
    j_validation = []
    train_set = train
    batcher = Batcher.Batcher(train[0], train[1])
    for training_percentage in range(10, 101, 10):
        with MultilayerPerceptron(input_size, neurons, output_size) as neural_net:
            # train_set = batcher.next_batch(int(len(train_set[0]) * (training_percentage / 100)))
            neural_net.train(train_set, alpha_rate, validation, epochs)
            j_train.append(neural_net.j(train[0], train[1]))
            j_validation.append(neural_net.j(validation[0], validation[1]))
            # x_axis = list(range(len(j_train)))
            x_axis.append(training_percentage)
            plt.plot(x_axis, j_train, 'r-', x_axis, j_validation, 'b-')
            plt.pause(0.00001)
    input("Konec analýzy pro pokračování stiskněte jakoukoliv klávesu")


if __name__ == '__main__':
    import utils.unzip
    from captcha_breaker import interactive

    # utils.unzip.unzip_datasets()
    input_data = []
    labels = []
    class_number = 0

    alphabet = "alphabet_13"
    samples_limit = 100
    # characters_limit = ['a','b','c']
    characters_limit = []

    print("Načítám data")
    for size in ["lowercase"]:
        for folder in os.listdir("./resources/output/%s/%s" % (alphabet, size)):
            if len(characters_limit) is 0 or folder in characters_limit:
                labels.append(folder)
                for i, sample in enumerate(os.listdir("./resources/output/%s/%s/%s" % (alphabet, size, folder))):
                    if i is samples_limit:
                        break
                    input_data.append({
                        "input": image_to_vector("./resources/output/%s/%s/%s/%s" % (alphabet, size, folder, sample)),
                        "output": class_number,
                        "class": folder
                    })
                class_number += 1
    print("Načetl jsem %s obrázků" % (len(input_data)))
    print("Počet labelů %s" % (len(labels)))
    print(labels)
    random.shuffle(input_data)

    train, validation, test = split(input_data, 60 / 100, 50 / 100)
    # train, validation, test = split(input_data, 75 / 100, 50 / 100)

    # plot_data(train, validation, test, labels)

    naive_test = test
    train = prepare_for_neural_network(train)
    validation = prepare_for_neural_network(validation)
    test = prepare_for_neural_network(test)

    # NEURAL NETWORK SETUP
    hidden_layer_neurons = 150
    learing_rate = 0.004
    epochs = 3000

    # dummies
    neural_network = dump_train(32 * 32, len(labels), train, validation, test, hidden_layer_neurons, learing_rate, epochs)
    accuracy_test = naive_accuracy_test(neural_network, naive_test, False)
    print("Přenost dle tensorflow: ", neural_network.accuracy(test)," %")


    # naive_accuracy_test_from_drive(neural_network, alphabet, 1000,False)

    # bias_variance_plot_debug(32 * 32, len(labels), train, validation, test, hidden_layer_neurons, learing_rate, epochs)
    # lambda_plot(32 * 32, hidden_layer_neurons, len(labels), train, validation, test, epochs, learing_rate, 0.001, 0.03)
    # epoch_plot(32 * 32, hidden_layer_neurons, len(labels), train, validation, test, learing_rate, 10000)
    # neural_plot(32 * 32, len(labels), train, validation, epochs, learing_rate, 10, 10, 1220)

