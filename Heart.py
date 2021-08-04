# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:35:09 2021

@author: Jose Miguel
"""

"""
Predecir Enfermedades Cardiovasculares
Proyecto Final de Inteligencia Artificial
"""

# Importar libreria basica para el proceso
import pandas as pd

# Importando los Datos de nuestra csv 
data = pd.read_csv('dataHeart.csv')
data.head()

# Colocar nombres a las columnas para que no se vean como
# V1 V2 V3 V4 V5 V6 V7 V8 V9, y tengan un formato mas limpio
# para asi poder analizar los datos de manera correcta
columnas = ['sbp', 'Tabaco', 'ldl', 'Adiposidad', 'Familia', 'Tipo', 'Obesidad', 'Alcohol', 'Edad', 'chd']

# Actualizando datos
data.columns = columnas
data.head()

# Analizaremos los datos, para asi poder conocer el formato de los datos
data.types

#Conoceremos los datos nulos
data.isnull().sum()

                    ### PROCESAMIENTO DE LOS DATOS ###
                    
# Cambiar los datos de Familia y CHD en digitales
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['Familia']=encoder.fit_transform(data['Familia'])
data['chd']=encoder.fit_transform(data['chd'])
data.head()

# Escalamos los valores de la columna sbp
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range =(0,100))
data['sbp'] = scale.fit_transform(data['sbp'].values.reshape(-1,1))
data.head()

                    ### VISUALIZACION DE LOS DATOS ###
#Visualizar la obesidad de acuerdo a la edad
data.plot(x='Edad',y='Obesidad',kind='scatter',figsize =(10,5))

#Visualizar el consumo de tabaco de acuerdo a la edad
data.plot(x='Edad',y='Tabaco',kind='scatter',figsize =(10,5))

#Visualizar el consumo de alcohol de acuerdo a la edad
data.plot(x='Edad',y='Alcohol',kind='scatter',figsize =(10,5))

                    ### ANALISIS DE MACHINE LEARNING ###
                    
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score   

#Definir las variable dependiente e independientes
y = data['chd']
X = data.drop('chd', axis =1)   

#Separar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1) 

#Definir el algoritmo
algoritmo = svm.SVC(kernel ='linear')

#Entrenar el algoritmo
algoritmo.fit(X_train, y_train)

#Realizar una predicción
y_test_pred = algoritmo.predict(X_test)

#Se calcula la matriz de confusión
print(confusion_matrix(y_test, y_test_pred))

#Se calcula la exactitud y precisión del modelo
accuracy_score(y_test, y_test_pred)
precision_score(y_test, y_test_pred)
            