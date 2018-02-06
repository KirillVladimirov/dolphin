# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # колличество узлов в слоях
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # коэффициент обучения
        self.learning_rate = learning_rate
        # матрицы весовых коэфициентов
        self.wih = NeuralNetwork.__normal(self.hidden_nodes, self.input_nodes)
        self.woh = NeuralNetwork.__normal(self.output_nodes, self.hidden_nodes)
        # сигмоида как функция активации
        self.activation_function = lambda x: expit(x)

    def train(self):
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.woh, hidden_outputs)
        final_output = self.activation_function(final_inputs)

        return final_output

    @staticmethod
    def __random(n1, n2):
        return np.random.rand(n1, n2) - 0.5

    @staticmethod
    def __normal(n1, n2):
        return np.random.normal(0.0, pow(n1, -0.5), (n1, n2))
