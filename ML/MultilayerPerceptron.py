from pprint import pprint

import tensorflow as tf
import matplotlib.pyplot as plt
from ML.Batcher import Batcher
import os

def summary(tensor, name):
    """
    Zařídí, že se zobrazí různé náhledy pro tensor v tensorboardu
    :param tensor: tensor, který je potřeba zobrazit
    """
    tf.summary.scalar(name, tensor)
    tf.summary.histogram(name, tensor)


def current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m_%H_%M_%S")


class MultilayerPerceptron:
    """
    Třívrstvý vícevrsvý perceptron
    """

    def __init__(self, input_size, number_of_neurons, output_size, session = None):
        self.input_size = input_size
        # Vstupní vrstva velikost x * input_size
        with tf.name_scope("neuronka"):
            # placeholder - neco jako konstanta, None - libovolná velikost v tomto směru
            self.input_layer = tf.placeholder(tf.float32, [None, self.input_size])
            # Skrytá vrstva náhodně inicializovaná náhodně input_size * number_of_neurons
            self.hidden_layer = tf.Variable(tf.random_normal([self.input_size, number_of_neurons]), name='skryta_vrstva')
            self.output_size = output_size
            # Výstupní vrstva (ta kterou to proženu, abych měl výstup v určitém formátu) náhodně inicializovaná number_of_neurons * output_size (počet tříd do kterých klasifikujeme)
            self.output_layer = tf.Variable(tf.random_normal([number_of_neurons, self.output_size]))
            # Bias pro skrytou vrstvu
            self.bias_1 = tf.Variable(tf.random_normal([number_of_neurons]))
            # self.bias_1 = 1
            # Bias pro výstupní vrstvu
            self.bias_2 = tf.Variable(tf.random_normal([output_size]))
            # self.bias_2 = 1
        if session is None:
            self.session = tf.InteractiveSession()
        else:
            self.session = session

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        # OK
        # Výstup ze skryté vrstvy relu(input_layer * hidden_layer + bias)
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input_layer, self.hidden_layer), self.bias_1))
        # Výstup neuronky hidden_output * output_layer + bias_2
        prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2))
        # Vrátí index největší hodnoty z výstupu
        return self.session.run(tf.argmax(prediction, 1), feed_dict={self.input_layer: input_data})

    def j_tensor(self, input_data, y_data):
        y_ = tf.constant(y_data, name='predpokladana_klasifikace')
        input_tensor = tf.constant(input_data)
        hidden_output = tf.nn.relu(tf.add(tf.matmul(input_tensor, self.hidden_layer), self.bias_1))
        feed_forward = tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2)
        # Konec feed forwardu máme předpověď
        # Naše cost funkce
        cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=feed_forward))
        return cross_entropy

    def j(self, input_data, y_data):
        tensor = self.j_tensor(input_data, y_data)
        return self.session.run(tensor)

    def stop(self, evt):
        if evt.key == 'q':
            self.run = False

    def train(self, training_set, learning_rate, validation_data, epochs, plot):
        """
        Trénování neuronové sítě
        :param training_set: trénovací množina
        :param learning_rate: učící parametr
        :param epochs: počet učících epoch
        """
        # Pro předčasné zastavení učení
        self.run = True
        # Data, která má neuronka předpovídat
        y_ = tf.placeholder(tf.float32, [None, self.output_size], name='predpokladana_klasifikace')
        # Feed forward
        #  Relu = Re(ctified) L(inear) (U)nit
        #  0 if the input is negative, and the input itself if that input is 0 or positive. This specific add-on function (or better "activation function")
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input_layer, self.hidden_layer), self.bias_1))
        feed_forward = tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2)
        # Konec feed forwardu máme předpověď
        # Naše cost funkce
        #   onehot_labels - co to ma predpovedet, logits - co to predpovida
        cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=feed_forward))
        # Samotná učení na jeden řádek dám mu learning rate a řeknu mu minimalizuj cost funkci a on to uděla :)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        saver = tf.train.Saver()
        # Přesnost na validačních datech
        tf.global_variables_initializer().run()
        # Batcher vrací náhodně promíchané vzorky
        batcher = Batcher(training_set[0], training_set[1])
        # Neuronku učíme po x epoch
        import matplotlib.pyplot as plt
        if plot:
            plt.ion()
            # Při uzavření vykreslovacího okna zastavíme učení
            plt.connect('key_press_event', self.stop)
            plt.ylabel("J($\\theta$)")
            plt.xlabel("Počet epoch * 10")
        j_hist = []
        j_validation_tensor = self.j_tensor(validation_data[0], validation_data[1])
        for i in range(epochs):
            training_set = batcher.next_batch(len(training_set))
            _, j = self.session.run([train_step, j_validation_tensor], feed_dict={self.input_layer: training_set[0], y_: training_set[1]})
            if i % 10 == 0 and plot:
                j_hist.append(j)
                plt.title("Učící křivka")
                plt.plot(j_hist, "r-")
                plt.pause(0.0000001)
            if not self.run:
                break
        if plot:
            print("Minmum J = ", j_hist[len(j_hist) - 1])
            # input("Stiskente kvalesu")
        # input("Stiskentě pro ukončení")
        # Uložení naučené neuronky
        # import os
        # time_stamp = current_time()
        # path = os.path.join(os.getcwd(), time_stamp)
        # os.mkdir(path)
        # saver.save(self.session, os.path.join(path, "model.ckpt"))

    def dump_train(self, training_set, learning_rate, validation_data, epochs):
        """
        Trénování neuronové sítě
        :param training_set: trénovací množina
        :param learning_rate: učící parametr
        :param epochs: počet učících epoch
        """
        # Pro předčasné zastavení učení
        self.run = True
        # Data, která má neuronka předpovídat
        y_ = tf.placeholder(tf.float32, [None, self.output_size], name='predpokladana_klasifikace')
        # Feed forward
        #  Relu = Re(ctified) L(inear) (U)nit
        #  0 if the input is negative, and the input itself if that input is 0 or positive. This specific add-on function (or better "activation function")
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input_layer, self.hidden_layer), self.bias_1))
        feed_forward = tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2)
        # Konec feed forwardu máme předpověď
        # Naše cost funkce
        #   onehot_labels - co to ma predpovedet, logits - co to predpovida
        cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=feed_forward))
        # Samotná učení na jeden řádek dám mu learning rate a řeknu mu minimalizuj cost funkci a on to uděla :)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        saver = tf.train.Saver()
        # Přesnost na validačních datech
        tf.global_variables_initializer().run()
        # Batcher vrací náhodně promíchané vzorky
        batcher = Batcher(training_set[0], training_set[1])
        # Neuronku učíme po x epoch
        j_hist = []
        j_validation_tensor = self.j_tensor(validation_data[0], validation_data[1])
        for i in range(epochs):
            training_set = batcher.next_batch(len(training_set))
            _, j = self.session.run([train_step, j_validation_tensor], feed_dict={self.input_layer: training_set[0], y_: training_set[1]})


    def accuracy_tensor(self, test_data):
        """
        Vrátí tensor, který dává přesnost vstupních dat po prohnání neuronkou
        """
        input_layer = tf.constant(test_data[0])
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer, self.hidden_layer), self.bias_1))
        prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, self.output_layer), self.bias_2))
        feed = tf.argmax(prediction, 1)
        comparsion_data = tf.constant(test_data[1])
        comparsion = tf.equal(feed, tf.argmax(comparsion_data, 1))
        return tf.reduce_mean(tf.cast(comparsion, tf.float32)) * 100

    def accuracy(self, test_data):
        # OK
        return self.session.run(self.accuracy_tensor(test_data), feed_dict={self.input_layer: test_data[0]})
    
    def error(self, test_data):
        # OK
        return 100 - self.accuracy(test_data)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
