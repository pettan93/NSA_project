#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * 2

if __name__ == '__main__':
    pozice = np.matrix(list(range(-100, 100)), dtype=np.float32).T
    data = f(pozice)
    with tf.Session() as sess:
        a = tf.Variable(0.0)
        b = tf.Variable(0.0)
        vstupni_data = tf.placeholder(tf.float32, [None, 1])
        hypoteza = a * vstupni_data + b
        realna_data = tf.placeholder(tf.float32, [None, 1])
        cost_funkce = tf.reduce_mean(tf.square(hypoteza - realna_data))
        trenovani = tf.train.GradientDescentOptimizer(0.01).minimize(cost_funkce)
        tf.initialize_all_variables().run()
        hodnoty = []
        for i in range(10000):
            _, hodnota_cost_funkce = sess.run([trenovani, cost_funkce], feed_dict={vstupni_data:pozice, realna_data:data})
            hodnoty.append(hodnota_cost_funkce)
        plt.plot(hodnoty, 'b--')
        plt.show()

        print("Zkousim na jinych datech")
        for x in range(-5, 6):
            vysledek = sess.run(hypoteza, feed_dict = {vstupni_data: np.matrix([float(x)])})
            print("f(%s) = %s" % (x, vysledek)) 
            
    
