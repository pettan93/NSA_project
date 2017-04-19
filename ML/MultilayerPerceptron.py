import tensorflow as tf
from ML.Batcher import Batcher

def summary(tensor, name):
    """
    Zařídí, že se zobrazí různé náhledy pro tensor v tensorboardu
    :param tensor: tensor, který je potřeba zobrazit
    """
    tf.summary.scalar(name, tensor)


def current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m_%H_%M_%S")


class MultilayerPerceptron:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, input_size, number_of_neurons, output_size):
        self.input_size = input_size
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_size])
        self.hidden_layer = tf.Variable(tf.random_normal([self.input_size, number_of_neurons]), name='skryta_vrstva')
        self.output_size = output_size
        self.output_layer = tf.Variable(tf.random_normal([number_of_neurons, self.output_size]))
        self.bias_1 = tf.Variable(tf.random_normal([number_of_neurons]))
        self.bias_2 = tf.Variable(tf.random_normal([output_size]))
        self.session = tf.InteractiveSession()

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input_layer, self.hidden_layer), self.bias_1))
        prediction = tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2)
        return self.session.run(tf.argmax(prediction, 1), feed_dict={self.input_layer: input_data})

    def train(self, training_set, learning_rate, epochs):
        """
        Trénování neuronové sítě
        :param training_set: trénovací množina
        :param learning_rate: učící parametr
        :param epochs: počet učících epoch
        """
        y_ = tf.placeholder(tf.float32, [None, self.output_size], name='predpokladana_klasifikace')
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input_layer, self.hidden_layer), self.bias_1))
        feed_forward = tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=feed_forward))
        summary(cross_entropy, "cost_funkce")
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        all_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./log/%s" % current_time(), self.session.graph)
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        for i in range(epochs):
            _, all = self.session.run([train_step, all_summaries],
                                      feed_dict={self.input_layer: training_set[0], y_: training_set[1]})
            train_writer.add_summary(all, i)
        import os
        time_stamp = current_time()
        path = os.path.join(os.getcwd(), time_stamp)
        os.mkdir(path)
        saver.save(self.session, os.path.join(path, "model.ckpt"))

    def accuracy(self, test_data):
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input_layer, self.hidden_layer), self.bias_1))
        prediction = tf.add(tf.matmul(hidden_output, self.output_layer), self.bias_2)
        feed_forward = tf.argmax(prediction, 1)
        comparsion_data = tf.constant(test_data[1])
        comparsion = tf.equal(feed_forward, tf.argmax(comparsion_data, 1))
        accuracy = tf.reduce_mean(tf.cast(comparsion, tf.float32))
        return self.session.run(accuracy, feed_dict={self.input_layer: test_data[0]}) * 100

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
