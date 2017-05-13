import random
from pprint import pprint
from subprocess import check_output
import os
import subprocess

import numpy
from Tools.scripts.treesync import raw_input
from matplotlib.pyplot import imshow

from ML.MultilayerPerceptron import MultilayerPerceptron
from PIL import Image, ImageOps
# from skimage.feature import corner_harris, corner_subpix, corner_peaks
import numpy as np

from main import image_to_vector


class Box:
    def __init__(self, x, y, w, h):
        self.width = w
        self.height = h
        self.x = x
        self.y = y
        self.w_x = self.x + w
        self.h_y = self.y + y

    def tuple(self):
        self.w_x = self.x + self.width
        self.h_y = self.y + self.height
        return (self.x, self.y, self.w_x, self.h_y)

    def inc_left(self):
        self.x += 1
        self.w_x = self.x + self.width

    def inc_top(self):
        self.y += 1
        self.h_y = self.y + self.height


def remove_borders(matrix, tol=200):
    mask = matrix < tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return matrix[x0:x1, y0:y1]


import numpy as np


def break_captcha(path):
    image = Image.open(path)
    print("Neuronka chce přečíst násedující obrázek..")
    image.show()

    box = Box(0, 0, 32, 32)
    # labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    #  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
    with MultilayerPerceptron(32 * 32, 200, len(labels)) as neural_network:
        neural_network.load("./2017-05_21_01_02/model.ckpt")
        while box.w_x < image.size[0] - 32:
            img_data = image.crop(box.tuple()).convert("LA").resize((32, 32))
            data = [x[0] for x in img_data.getdata()]
            matrix = np.matrix(data).reshape((32, 32))
            matrix = remove_borders(matrix)
            if matrix is None:
                box.x += 20
                continue
            array = np.asarray(matrix, dtype=np.uint8)
            data_img = Image.fromarray(array)
            if data_img.size[0] < 10 or data_img.size[1] < 10:
                box.x += 20
                continue
            data_img = ImageOps.invert(data_img).resize((32, 32)).convert("LA")
            data_img.show()
            data = np.matrix([x[0] / 255 for x in data_img.getdata()])

            indx = neural_network.feed_forward(data)
            print("Neuronka vyhodnotila [" + labels[indx[0]] + "]")
            box.x += box.width
            input("enter pro další znak")

"""
Interaktivni mod pro zkoušení klasifikace
"""
def interactive(neural_network, alphabet_number,labels,sample_number=False):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    input_char = raw_input("Jaký znak chcete klasifikovat?: ").strip().rstrip('\n')

    total = 0
    success = 0

    while True:
        while not (input_char.isalpha() and len(input_char) is 1):
            print("Není v datech, zkuste znovu.")
            input_char = raw_input("Jaký znak chcete klasifikovat?: ").strip().rstrip('\n')

        if sample_number is False:
            path = random.choice(
                os.listdir("./resources/output/%s/lowercase/%s" % (alphabet_number, input_char)))
            path = "./resources/output/%s/lowercase/%s/" % (alphabet_number, input_char) + path
            print("Nahodne vybrany vzorek znaku [" + input_char + "] - otevíram..")
        else:
            input_sample = raw_input("Cislo vzorku?: ").strip().rstrip('\n')
            path = "./resources/output/%s/lowercase/%s/%s.png" % (alphabet_number, input_char,input_sample)

        plt.imshow(mpimg.imread(path))
        plt.show(block=False)

        image_vector = image_to_vector(path)

        data = np.array(image_vector).reshape(1, 1024)

        indx = neural_network.feed_forward(data)

        print(" => Neuronka vyhodnotila : [" + labels[indx[0]] + "]\n")
        total += 1
        if labels[indx[0]] is input_char:
            success += 1
        print("Úspěšnost [" + str(success) + "/" + str(total) + "] - " + str((success / total) * 100) + " %")

        input_char = raw_input("Jaký znak chcete klasifikovat?: ").strip().rstrip('\n')

        plt.close()




if __name__ == '__main__':
    # break_captcha("resources/output/alphabet_3/captcha3.png")

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
    with MultilayerPerceptron(32 * 32, 200, len(labels)) as neural_network:
        neural_network.load("./2017-05_21_01_02/model.ckpt")
        interactive(neural_network, 13)
