# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:34:05 2024

@author: niqui
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
import numpy as np


#%% Cargar los datos
pixeles_completo = pd.read_csv('C:/Users/niqui/Documents/Labo de Datos/TP2_lab_datos_2024-/Datos/TMNIST_Data.csv')

# Quitar la fuente 
pixeles = pixeles_completo.iloc[:, 1:]

#%% Filtrar el dataset de forma que incluya las filas donde las columnas sean 0 o 1
subconjunto = pixeles[pixeles['labels'].isin([0, 1])]

# Cuenta la cantidad de ocurrencias de cada clase (0 y 1) en el dataset filtrado
contador_de_clase = subconjunto['labels'].value_counts()

# Mostrar el número de muestras en el subconjunto y si está balanceado
balanceado= (contador_de_clase[0] == contador_de_clase[1])
 
print(subconjunto.shape[0], contador_de_clase,)
print(  'Balanceado:', balanceado)

#%% 
# Separar características (X) y etiquetas (y)
X = subconjunto.drop(columns=['labels'])
Y = subconjunto['labels']

# Dividir los datos en conjuntos de entrenamiento y prueba (70% para train y 30% para test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Definir algunos subconjuntos de 3 atributos a probar
combinacion_atributos = [
    ['10', '245', '355'], 
    ['50', '100', '150'], 
    ['400', '650', '234']
]

# Variables para almacenar los resultados
precisions = []
recalls = []
accuracy_scores = []
confusion_matrices = []

# Calcular métricas para cada subconjunto de atributos
for atributo in combinacion_atributos:
    # Definir subconjunto de entrenamiento y prueba
    X_train_subconjunto = X_train[atributo]
    X_test_subconjunto = X_test[atributo]

    # Entrenar el modelo con el subconjunto de atributos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subconjunto, Y_train)
    Y_pred = model.predict(X_test_subconjunto)

    # Calcular métricas y almacenar resultados
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='binary')
    recall = recall_score(Y_test, Y_pred, average='binary')
    cm = confusion_matrix(Y_test, Y_pred)

    # Almacenar resultados
    accuracy_scores.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    confusion_matrices.append(cm)

    # Imprimir resultados para cada combinación
    print(f"Atributos: {atributo}")
    print(f"Exactitud: {accuracy}")
    print(f"Precisión: {precision}")
    print(f"Exhaustividad: {recall}")
    print(f"Matriz de Confusión:\n{cm}\n\n")

# 1. Gráfico de barras para la exactitud
plt.figure(figsize=(10, 6))
plt.bar([' - '.join(atr) for atr in combinacion_atributos], accuracy_scores, color='skyblue')
plt.xlabel('Subconjunto de Atributos', fontsize = 14)
plt.ylabel('Exactitud', fontsize = 14)
plt.title('Exactitud de KNN para diferentes subconjuntos de atributos', fontsize = 18)
plt.xticks(rotation=45)
plt.show()

# 2. Matriz de confusión para cada subconjunto de atributos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (atr, cm) in enumerate(zip(combinacion_atributos, confusion_matrices)):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Matriz de Confusión para atributos {' - '.join(atr)}", fontsize = 14)
    axes[i].set_xlabel('Predicción', fontsize = 14)
    axes[i].set_ylabel('Real', fontsize = 14)

plt.tight_layout()
plt.show()

# 3. Gráfico de barras apiladas para precisión y exhaustividad
x = np.arange(len(combinacion_atributos))
width = 0.35  # Ancho de las barras

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, precisions, width, label='Precisión')
rects2 = ax.bar(x + width/2, recalls, width, label='Exhaustividad')

ax.set_xlabel('Subconjunto de Atributos', fontsize = 14)
ax.set_ylabel('Valores', fontsize = 14)
ax.set_title('Precisión y Exhaustividad de KNN para diferentes subconjuntos de atributos', fontsize = 18)
ax.set_xticks(x)
ax.set_xticklabels([' - '.join(atr) for atr in combinacion_atributos], rotation=45)
ax.legend()

plt.show()


#%% 

X = subconjunto.drop(columns=['labels'])
Y = subconjunto['labels']
      
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test

# Definir algunos subconjuntos de 5 atributos a probar
combinacion_atributos = [
    ['10', '245', '355','478','505'], 
    ['50', '100', '150','200','250'], 
    ['400', '450', '500','550','600']
]

results = []

for atributo in combinacion_atributos:
    X_train_subconjunto = X_train[atributo]
    X_test_subconjunto = X_test[atributo]

    # Entrenar el modelo con el subconjunto de atributos
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_subconjunto, Y_train)
    Y_pred = model.predict(X_test_subconjunto)

    # Calcular metricas
    accuracy = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    results.append((atributo, accuracy, cm))
    precision = precision_score(Y_test, Y_pred, average='binary')
    recall = recall_score(Y_test, Y_pred, average='binary')

    print(f"Atributos: {atributo}")
    print(f"Exactitud: {accuracy}")
    print(f"Precisión: {precision}")
    print(f"Exhaustividad: {recall}")
    print(f"Matriz de Confusión:\n{cm}\n\n")


#%%

X = subconjunto.drop(columns=['labels'])
Y = subconjunto['labels']
      
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test

# Definir algunos subconjuntos de atributos para probar
combinacion_atributos = [
    ['10', '245', '355'], 
    ['50', '100', '150'], 
    ['400', '450', '500']
]

# Definir valores de k a probar
valores = [1, 3, 5, 7, 10]

results = []

# Probar cada combinación de atributos y cada valor de k
for atributos in combinacion_atributos:
    for k in valores:
        # Seleccionar los atributos específicos para el conjunto de entrenamiento y prueba
        X_train_subconjunto = X_train[atributos]
        X_test_subconjunto = X_test[atributos]
        
        # Crear y entrenar el modelo KNN
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_subconjunto, Y_train)
        
        # Predecir sobre el conjunto de prueba
        Y_pred = model.predict(X_test_subconjunto)
        
        # Calcular exactitud
        accuracy = accuracy_score(Y_test, Y_pred)
        
        results.append({
            'Atributos': ', '.join(atributos),
            'k': k,
            'Exactitud': accuracy
        })

results_df = pd.DataFrame(results)

# Convertir los resultados en formato adecuado para el heatmap
heatmap_data = results_df.pivot(index="Atributos", columns="k", values="Exactitud")

# Dibujar el heatmap
plt.figure(figsize=(12, 6))
heatmap = sns.heatmap(
    heatmap_data, 
    annot=True, 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Exactitud'}
)

# Ajustar el tamaño de la fuente de la etiqueta de la barra de color
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Exactitud', fontsize=14)  # Ajusta el tamaño de fuente aquí

# Títulos y etiquetas de los ejes
plt.title('Exactitud de KNN para distintos atributos y valores de k', fontsize=18)
plt.xlabel('Número de vecinos (k)', fontsize=14)
plt.ylabel('Conjunto de atributos', fontsize=14)

plt.show()
