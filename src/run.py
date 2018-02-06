# -*- coding: utf-8 -*-

from dolphin.models import NeuralNetwork


if __name__ == "__main__":
    nn = NeuralNetwork(3, 3, 3, 0.3)
    print(nn.wih)
