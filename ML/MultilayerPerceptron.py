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
    def __init__(self, input_size, output_size):
        self.hidden_layer = tf.Variable(tf.random_normal([input_size, output_size]), name='skryta_vrstva')
        self.b = tf.Variable(tf.zeros([output_size]), name='bias')
        self.input_size = input_size
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_size])
        self.output_size = output_size
        self.session = tf.InteractiveSession()

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        feed_forward = tf.argmax(tf.nn.softmax(tf.matmul(self.input_layer, self.hidden_layer) + self.b), 1)
        return self.session.run(feed_forward, feed_dict={self.input_layer: input_data})

    def train(self, training_set, learning_rate, epochs):
        """
        Trénování neuronové sítě
        :param training_set: trénovací množina
        :param learning_rate: učící parametr
        :param epochs: počet učících epoch
        """
        batcher = Batcher(training_set[0], training_set[1])
        y_ = tf.placeholder(tf.float32, [None, self.output_size], name='predpokladana_klasifikace')
        feed_forward = tf.nn.softmax(tf.matmul(self.input_layer, self.hidden_layer) + self.b, name='predikce')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=feed_forward))
        summary(cross_entropy, "cost_funkce")
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        all_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./log/%s" % current_time(), self.session.graph)
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        for i in range(epochs):
            train_data = batcher.next_batch(100)
            _, all = self.session.run([train_step, all_summaries],
                                      feed_dict={self.input_layer: train_data[0], y_: train_data[1]})
            train_writer.add_summary(all, i)
        import os
        time_stamp = current_time()
        os.mkdir("./save/%s" % time_stamp)
        saver.save(self.session, "./save/%s/model.ckpt" % time_stamp)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
