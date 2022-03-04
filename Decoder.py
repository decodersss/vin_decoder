# -*- coding: utf-8 -*-
"""Vin Decoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VwmIbU5K3KdRaA5beL9i6wQxmOPCx9v2
"""

#Importações das bibliotecas
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

#Caminho do arquivo
path = "/content/AudiDenatran.csv"

#Criando o dataframe utilizando Pandas
dataframe = pd.read_csv(path, sep=';')

#Criando a variavel HEAD_DATA
head_data = dataframe;

#"Splitagem" da base a partir do chassi
X_data = head_data['Chassi'].str.split(r"", expand=True).drop([0, 18], axis = 1)

#Transformando as colunas em binario (Matriz) - Para saber a posição
x_data_encoded = OneHotEncoder().fit_transform(X_data)

X_data.head()

print(x_data_encoded)

#Selecionando as informações unicas de cada label
modelo_mask = dataframe.Modelo.unique()
versao_mask = dataframe.Versao.unique()
ano_mask = dataframe.AnoModel.unique()

#Exemplo
versao_mask[0]

#Utilizando Numpy para transformar as labels em array
modelLabels = np.array(dataframe[['Modelo']].astype('category'))
versionBin = pd.get_dummies(dataframe[['Versao']].astype('category'))
versionLabels = np.array(versionBin)

yearLabels = np.array(dataframe[['AnoModel']].astype('category'))

print(versionLabels)

#Transformando a coluna Y em binario 
modelBinary = LabelBinarizer()
versionBinary = LabelBinarizer()
yearBinary = LabelBinarizer()

modelLabels = modelBinary.fit_transform(modelLabels)
versionLabels = versionBinary.fit_transform(versionLabels)
yearLabels = yearBinary.fit_transform(yearLabels)

#Print do modelo binario
yearLabels

#função na seleção de modelos Sklearn para dividir matrizes de dados em dois subconjuntos: para dados de treinamento e para dados de teste.
split = train_test_split(x_data_encoded, modelLabels, versionLabels, yearLabels, test_size=0.2, random_state=42)
(trainX, testX, trainmodelY, testmodelY, trainversionY, testversionY, trainyearY, testyearY) = split

#Abertura do neuronio
models = keras.Sequential()
#Entrada do neuronio. Abertura
models.add(keras.Input(shape=(171,)))
#Arquitetura de modelo. "Camada" Forma de somar neuronios. Soma de positivos.
models.add(layers.Dense(64, activation="relu"))
models.add(layers.Dense(len(dataframe.Modelo.unique()), activation = "softmax", name='modelo_output'))
models.add(layers.Dense(len(dataframe.Versao.unique()), activation = "softmax", name='versao_output'))
models.summary()

Inputs = keras.Input(trainX.shape[1], name="input_modelo")
Features = layers.Dense(64, activation="relu")(Inputs)

Outputs = layers.Dense(10, activation="softmax")(Features)

ModelsOutputs = layers.Dense(len(dataframe.Modelo.unique()), activation="softmax", name='modelOutput')(Features)
VersionsOutputs = layers.Dense(89, activation="softmax", name='versionsOutput')(Features)

model = keras.Model(inputs=Inputs, outputs=[ModelsOutputs, VersionsOutputs])
model.summary()

model.compile(optimizer="rmsprop",loss="binary_crossentropy", metrics=["accuracy"])

losses = {
    "modelOutput": "binary_crossentropy",
    "versionsOutput": "binary_crossentropy"
}

lossweights = {
    "modelOutput":1.0,
    "versionsOutput":1.0
}

metrics = {
    "modelOutput": "accuracy",
    "versionsOutput": "accuracy"
}

model.compile(optimizer="rmsprop", loss=losses, loss_weights=lossweights, metrics=metrics)

ep = 50
bt = 512

history = model.fit(x=trainX,
          y={"modelOutput": trainmodelY,
             "versionsOutput": trainversionY},
                    validation_data=(testX, 
                                     {"modelOutput": testmodelY,
                                      "versionsOutput": testversionY}),
                    epochs=ep, batch_size=bt, verbose=1)

vinMask = pd.get_dummies(X_data).columns

vinMask[[0,1,2,9,17,36,51,59,79,89,110,116,129,135,150,157,161]]

X_data

model.predict(testX[150])

prev_model = np.argmax(model.predict(testX[150])[0])
prev_version = np.argmax(model.predict(testX[150])[1])

modelo_mask[prev_model]

versao_mask[prev_version]

x = 380

prevs_model = np.argmax(model.predict(testX[x])[0])
prevs_version = np.argmax(model.predict(testX[x])[1])

print('Modelo: ' + modelo_mask[prevs_model],
      'Versão: ' + versao_mask[prevs_version])

import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask
from flask import request
from flask import jsonify, make_response
app = Flask(__name__)
path = "/content/AudiDenatran.csv"
#Criando o dataframe utilizando Pandas
dataframe = pd.read_csv(path, sep=';')
#Split dos caracteres do chassi em um array
X_data = dataframe['Chassi'].str.split(r"", expand=True).drop([0, 18], axis = 1)
#Transformando as colunas em binario (Matriz)
#aqui vamos salvar a instancia do OneHotEncoder
#para poder utilizar ele para codificar uma string
#que vier na chamada de API do flask
encoder = OneHotEncoder();
x_data_encoded = encoder.fit_transform(X_data);
#Selecionando as informações unicas de cada label
modelo_mask = dataframe.Modelo.unique().astype(str)
versao_mask = dataframe.Versao.unique().astype(str)
ano_mask = dataframe.AnoModel.unique().astype(str)
#ordenando os itens unicos das colunas de label
modelo_mask.sort()
versao_mask.sort()
ano_mask.sort()
#Utilizando Numpy para transformar as labels em array
modelLabels = np.array(dataframe[['Modelo']].astype('str'))
versionLabels = np.array(dataframe[['Versao']].astype('str'))
yearLabels = np.array(dataframe[['AnoModel']].astype('str'))
#Transformando as colunas Y (labels) em binario 
modelBinarizer = LabelBinarizer()
versionBinarizer = LabelBinarizer()
yearBinarizer = LabelBinarizer()
modelLabels = modelBinarizer.fit_transform(modelLabels)
versionLabels = versionBinarizer.fit_transform(versionLabels)
yearLabels = yearBinarizer.fit_transform(yearLabels)
#preparação dos dados usados no modelo
Inputs = keras.Input(x_data_encoded.shape[1], name="input_modelo")
Features = layers.Dense(128, activation="relu")(Inputs)
ModelsOutputs = layers.Dense(len(modelo_mask), activation="softmax", name='modelOutput')(Features)
VersionsOutputs = layers.Dense(len(versao_mask), activation="softmax", name='versionsOutput')(Features)
YearsOutputs = layers.Dense(len(ano_mask), activation="softmax", name='yearsOutput')(Features)
#criação do modelo
model = keras.Model(inputs=Inputs, outputs=[ModelsOutputs, VersionsOutputs, YearsOutputs])
losses = {
    "modelOutput": "binary_crossentropy",
    "versionsOutput": "binary_crossentropy",
    "yearsOutput": "binary_crossentropy"
}
lossweights = {
    "modelOutput":1.0,
    "versionsOutput":1.0,
    "yearsOutput":1.0
}
metrics = {
    "modelOutput": "accuracy",
    "versionsOutput": "accuracy",
    "yearsOutput": "accuracy"
}
# compilação do modelo
model.compile(optimizer="rmsprop", loss=losses, loss_weights=lossweights, metrics=metrics)
ep = 80
bt = 500
# execução do treinamento do modelo
history = model.fit(x=x_data_encoded,
          y={"modelOutput": modelLabels,
             "versionsOutput": versionLabels,
             "yearsOutput": yearLabels},
          epochs=ep, batch_size=bt, verbose=1)
# Treinamento finalizado. Vamos publicar uma API para testar o modelo!
@app.route("/decode")
def get_vin():
    vin = request.args.get('vin');
    print(vin)
    # reutilização do OneHotEncoder para obter a matriz esparsa do VIN codificado para usar o predict do modelo.
    matrix = encoder.transform([list(vin)])
    prev_model = np.argmax(model.predict(matrix)[0])
    prev_version = np.argmax(model.predict(matrix)[1])
    prev_year = np.argmax(model.predict(matrix)[2])
    # montagem do JSON e obtenção do label em formato string usando os dicionarios.
    json = '{"maker":"Audi", "model":"'+modelo_mask[prev_model]+'", "version":"'+versao_mask[prev_version]+'","year":"'+ano_mask[prev_year]+'"}'
    return json

model = keras.Model(inputs=Inputs, outputs=[ModelsOutputs, VersionsOutputs, YearsOutputs])
losses = {
    "modelOutput": "binary_crossentropy",
    "versionsOutput": "binary_crossentropy",
    "yearsOutput": "binary_crossentropy"
}
lossweights = {
    "modelOutput":1.0,
    "versionsOutput":1.0,
    "yearsOutput":1.0
}
metrics = {
    "modelOutput": "accuracy",
    "versionsOutput": "accuracy",
    "yearsOutput": "accuracy"
}

Inputs = keras.Input(x_data_encoded.shape[1], name="input_modelo")
Features = layers.Dense(128, activation="relu")(Inputs)

print(Inputs)

Inputs = keras.Input(x_data_encoded.shape[1], name="input_modelo")
print(keras.Input(x_data_encoded.shape[1]))

print(Features)

ModelsOutputs = layers.Dense(len(modelo_mask), activation="softmax", name='modelOutput')(Features)
VersionsOutputs = layers.Dense(len(versao_mask), activation="softmax", name='versionsOutput')(Features)
YearsOutputs = layers.Dense(len(ano_mask), activation="softmax", name='yearsOutput')(Features)

print(layers)

model.compile(optimizer="rmsprop", loss=losses, loss_weights=lossweights, metrics=metrics)
ep = 80
bt = 500

model = keras.Model(inputs=Inputs, outputs=[ModelsOutputs, VersionsOutputs, YearsOutputs])
losses = {
    "modelOutput": "binary_crossentropy",
    "versionsOutput": "binary_crossentropy",
    "yearsOutput": "binary_crossentropy"
}
lossweights = {
    "modelOutput":1.0,
    "versionsOutput":1.0,
    "yearsOutput":1.0
}
metrics = {
    "modelOutput": "accuracy",
    "versionsOutput": "accuracy",
    "yearsOutput": "accuracy"
}