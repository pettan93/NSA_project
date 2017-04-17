import tensorflow as tf

def summary(tensor, name):
    """
    Zařídí, že se zobrazí různé náhledy pro tensor v tensorboardu
    :param tensor: tensor, který je potřeba zobrazit
    """
    tf.summary.scalar(name, tensor)


class MultilayerPerceptron:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, input_size, output_size):
        self.hidden_layer = tf.Variable(tf.zeros([input_size, output_size]), name='skryta_vrstva')
        self.b = tf.Variable(tf.zeros([output_size]), name='bias')
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
        """
        Trénování neuronové sítě
        :param training_set: trénovací množina
        :param learning_rate: učící parametr
        :param epochs: počet učících epoch
        """
        x = tf.placeholder(tf.float32, [None, self.input_size], name='vstupni_vektor')
        y_ = tf.placeholder(tf.float32, [None, self.output_size], name='predpokladana_klasifikace')
        feed_forward = tf.nn.softmax(tf.matmul(x, self.hidden_layer) + self.b, name='predikce')
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(feed_forward), reduction_indices=[1]))
        summary(cross_entropy, "cost_funkce")
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(feed_forward, 1), tf.argmax(y_, 1)), tf.float32))
        summary(accuracy, "presnost")
        all_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./log", self.session.graph)
        tf.global_variables_initializer().run()
        for i in range(epochs):
            _, acc, all = self.session.run([train_step, accuracy, all_summaries], feed_dict={x: training_set[0], y_: training_set[1]})
            train_writer.add_summary(all, i)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
