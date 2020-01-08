"""
Classe utilizada para treinar a rede neural

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import pandas as pd
import keras
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils as np_utils
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)
def read_Dataset(path):
    base = pd.read_csv(path)
    print(path)
    # separa as características das classes
    #função iloc pega todas as colunas de 0 até 3, o limite superior não faz parte do intervalo
    caracteristicas= base.iloc[:, 0:12].values
    #pega apenas a última coluna, que contém as classes
    classe = base.iloc[:, 12].values

    #transforma as classes que estão em string para inteiros
    labelencoder = LabelEncoder()
    classe = labelencoder.fit_transform(classe)
    #cria classes falsas para cada neurônio de saída

    classe_dummy = np_utils.to_categorical(classe)
    # iris setosa     1 0 0
    # iris virginica  0 1 0
    # iris versicolor 0 0 1

    from sklearn.model_selection import train_test_split
    #divide a base
    caracteristicas_treinamento, caracteristicas_teste, classe_treinamento, classe_teste = train_test_split(caracteristicas, classe_dummy, test_size=0.25)
    classes = 3 #possíveis classes
    batch_size = 10
    epochs = 1000
    input_shape = 12 #quantidade de características
    return (classes, batch_size, input_shape,caracteristicas_treinamento, caracteristicas_teste, classe_treinamento, classe_teste)


def compile_model(network, nb_classes, input_shape):
    """Compila o modelo sequencial.

    Args:
        network (dict): parâmetros da rede

    Returns:
        a rede compilada.

    """
    # pega os parâmetros da rede.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Adiciona as layers separadamente
    for i in range(nb_layers):

        # para a primeira camada, a quantidade de neurônios é igual a quantidade de atributos.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_dim=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Camada de saída.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Treina o modelo e retorna a acurácia.

    Args:
        network (dict): parâmetros da rede
        dataset (str): Dataset usado para treino e teste

    """

    nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = read_Dataset(dataset)

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
