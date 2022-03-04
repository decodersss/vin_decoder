#Importações das bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask
from flask import request
from flask import jsonify, make_response
app = Flask(__name__)
path = "D:\AudiDenatran.csv"
# Criando o dataframe utilizando Pandas
dataframe = pd.read_csv(path, sep=';')
# Split dos caracteres do chassi em um array
X_data = dataframe['Chassi'].str.split(r"", expand=True).drop([0, 18], axis = 1)
# Transformando as colunas em binario (Matriz)
# aqui vamos salvar a instancia do OneHotEncoder
# para poder utilizar ele para codificar uma string
# que vier na chamada de API do flask
encoder = OneHotEncoder();
x_data_encoded = encoder.fit_transform(X_data);
# Selecionando as informações unicas de cada label
modelo_mask = dataframe.Modelo.unique().astype(str)
versao_mask = dataframe.Versao.unique().astype(str)
ano_mask = dataframe.AnoModelo.unique().astype(str)
# ordenando os itens unicos das colunas de label
modelo_mask.sort()
versao_mask.sort()
ano_mask.sort()
# Utilizando Numpy para transformar as labels em array
modelLabels = np.array(dataframe[['Modelo']].astype('str'))
versionLabels = np.array(dataframe[['Versao']].astype('str'))
yearLabels = np.array(dataframe[['AnoModelo']].astype('str'))
# Transformando as colunas Y (labels) em binario 
modelBinarizer = LabelBinarizer()
versionBinarizer = LabelBinarizer()
yearBinarizer = LabelBinarizer()
modelLabels = modelBinarizer.fit_transform(modelLabels)
versionLabels = versionBinarizer.fit_transform(versionLabels)
yearLabels = yearBinarizer.fit_transform(yearLabels)
# preparação dos dados usados no modelo
Inputs = keras.Input(x_data_encoded.shape[1], name="input_modelo")
Features = layers.Dense(128, activation="relu")(Inputs)
ModelsOutputs = layers.Dense(len(modelo_mask), activation="softmax", name='modelOutput')(Features)
VersionsOutputs = layers.Dense(len(versao_mask), activation="softmax", name='versionsOutput')(Features)
YearsOutputs = layers.Dense(len(ano_mask), activation="softmax", name='yearsOutput')(Features)
# criação do modelo
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
    # reutilização do OneHotEncoder para obter 
    # a matriz esparsa do VIN codificado
    # para usar o predict do modelo.
    matrix = encoder.transform([list(vin)])
    prev_model = np.argmax(model.predict(matrix)[0])
    prev_version = np.argmax(model.predict(matrix)[1])
    prev_year = np.argmax(model.predict(matrix)[2])
    # montagem do JSON e obtenção do label
    # em formato string usando os dicionarios.
    json = '{"maker":"Audi", "model":"'+modelo_mask[prev_model]+'", "version":"'+versao_mask[prev_version]+'","year":"'+ano_mask[prev_year]+'"}'
    return json
