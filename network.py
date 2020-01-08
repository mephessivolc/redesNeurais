"""Classe que representa uma rede neural"""
import random
import logging
from train import train_and_score

class Network():
    

    def __init__(self, nn_param_choices=None):
        """Inicializa a rede neural.

        Args:
            nn_param_choices (dict): Parâmetros da rede:
                neurônios por camada - nb_neurons (list): [64, 128, 256] 
                camadas - nb_layers (list): [1, 2, 3, 4]
                funções de ativação - activation (list): ['relu', 'elu']
                algoritmo de aprendizado  - optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {} 

    def create_random(self):
        """Função que cria uma rede neural aleatória"""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Configura as propriedades da rede.

        Args:
            network (dict): Parâmetros da rede

        """
        self.network = network

    def train(self, dataset):
        """Realiza o treinamento da rede e retorna a acurácia da mesma.

        Args:
            dataset (str): path do dataset (conjunto de teste)

        """
        print(dataset)
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset)

    def print_network(self):
        """Imprime o resultado da rede"""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
