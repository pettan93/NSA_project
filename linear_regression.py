#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

if __name__ == '__main__':
    pozice = np.array(list(range(-100, 100)), dtype=np.float32)
    data = f(pozice)
    print(pozice)
    with tf.Session() as sess:
        a = tf.Variable(0.0)
        b = tf.Variable(0.0)
        vstupni_data = tf.placeholder(tf.float32, [None])
        hypoteza = a * vstupni_data + b
        realna_data = tf.placeholder(tf.float32, [None])
        cost_funkce = tf.losses.mean_squared_error(realna_data, hypoteza)
        trenovani = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost_funkce)
        tf.initialize_all_variables().run()
        hodnoty = []
        for i in range(10000):
            _, hodnota_cost_funkce = sess.run([trenovani, cost_funkce], feed_dict={vstupni_data:pozice, realna_data:data})
            hodnoty.append(hodnota_cost_funkce)

        par_a, par_b = sess.run([a, b])
        print("Model %s * x + %s" % (par_a, par_b))
        plt.plot(hodnoty, 'b--')
        plt.show()
        print("Zkousim na jinych datech")
        x = list(range(-10, 10))
        vysledek = sess.run(hypoteza, feed_dict = {vstupni_data: np.array(x)})
        print("f(%s) =\n %s" % (x, vysledek))
        predpoved = sess.run([hypoteza], feed_dict={vstupni_data: pozice})
        plt.plot(pozice, data.tolist(), 'yo', pozice, predpoved[0], 'r-')
        plt.show()
            
    
