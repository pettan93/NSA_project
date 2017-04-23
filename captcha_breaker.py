from ML.MultilayerPerceptron import MultilayerPerceptron
from PIL import Image, ImageOps
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import numpy as np


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


def break_captcha(path):
    image = Image.open(path)
    box = Box(0, 8, 50, 50)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    neural_network = MultilayerPerceptron(100 * 100, 450, len(labels))
    neural_network.load("./10000_450_97/model.ckpt")
    while box.w_x < image.size[0] - 50:
        img_data = ImageOps.invert(image.crop(box.tuple())).resize((100, 100)).convert("LA")
        data = [x[0] for x in img_data.getdata()]
        data = np.matrix([x / 255 for x in data])
        indx = neural_network.feed_forward(data)
        print(labels[indx[0]], end="")
        box.x += 50


if __name__ == '__main__':
    break_captcha("resources/output/out1.png")
