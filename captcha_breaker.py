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

def remove_borders(matrix, tol = 150):
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
    box = Box(20, 0, 20, 32)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    with MultilayerPerceptron(32 * 32, 50, len(labels)) as neural_network:
        neural_network.load("./2017-04_09_45_42/model.ckpt")
        while box.w_x < image.size[0] - 20:
            img_data = image.crop(box.tuple()).convert("LA").resize((32, 32))
            data = [x[0] for x in img_data.getdata()]
            matrix = np.matrix(data).reshape((32, 32))
            matrix = remove_borders(matrix)
            if matrix is None:
                box.x += 20
                continue
            data_img = Image.fromarray(matrix)
            if data_img.size[0] < 10 or data_img.size[1] < 10:
                box.x += 20
                continue
            data_img = data_img.resize((32, 32)).convert("LA")
            data = np.matrix([(255 - x[0]) / 255 for x in data_img.getdata()])
            indx = neural_network.feed_forward(data)
            print(labels[indx[0]], end="")
            box.x += 20


if __name__ == '__main__':
    break_captcha("resources/output/captcha5.png")
