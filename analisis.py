# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:52:06 2024

@author: mgo20
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier , plot_tree

#%% Cargar los datos
pixeles_completo = pd.read_csv('C:/Users/mgo20/OneDrive/Desktop/Data/plab/TP2_lab_datos_2024-/Datos/TMNIST_Data.csv')
# Cargo el DataFrame de las matrices de cada combinación 
resultado_combinacion = pd.read_csv('C:/Users/mgo20/OneDrive/Desktop/Data/plab/TP2_lab_datos_2024-/Datos/resultados_combinaciones.csv')

#%% Quitar la fuente de pixeles
pixeles = pixeles_completo.iloc[:, 1:]

#%%######################

####   Pequeño analisis 

#########################
#%% Seleccionar una fila para mostrar
row = pixeles.iloc[21]  # Selecciona una fila específica

# Extraer la etiqueta
label = row[0]  # Primer elemento ahora es la etiqueta

# Extraer los valores de los píxeles
img_data = row[1:].values  
img = img_data.reshape((28, 28)) 

# Graficar la imagen
plt.imshow(img, cmap='gray')
plt.title(f"Dígito: {label}")  
plt.axis('off')  
plt.show()

#%% Información sobre los datos
pixeles.info()
pixeles.describe()
pixeles.head()

#%% Comprobar si los datos están balanceados
print(pixeles['labels'].value_counts())  # Sip, Los datos están balanceados

#%%######################

####   Ver Dev

#########################

#%% Dividir los datos en conjuntos de desarrollo (dev) y de prueba (held-out)
dev_pixeles, held_out_pixeles = train_test_split(pixeles, test_size=0.2, stratify=pixeles['labels'], random_state=42)

print("Distribución en dev:")
print(dev_pixeles['labels'].value_counts())
print("\nDistribución en held-out:")
print(held_out_pixeles['labels'].value_counts())

X = dev_pixeles.drop(columns=['labels'])
y = dev_pixeles['labels']

#%% Definir función para calcular métricas de evaluación
# La función que llama esto se comentó porque demora mucho crear los datos, se usó y almacenó los datos en un csv
def calcular_metricas(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))  # Verdaderos positivos
    fp = np.sum((y_pred == 1) & (y_true == 0))  # Falsos positivos
    fn = np.sum((y_pred == 0) & (y_true == 1))  # Falsos negativos
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

#%% Función para evaluar el modelo de árbol de decisión con combinaciones de hiperparámetros
# La función que llama esto se comentó porque demora mucho crear los datos, se usó y almacenó los datos en un csv
def evaluar_arbol(X, y, max_depth, min_samples_split, min_samples_leaf, criterion):
    modelo = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        criterion=criterion, 
        random_state=42
    )
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        precision, recall, f1 = calcular_metricas(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        resultados.append({
            'fold': fold + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion
        })

    return resultados

#%% Crea las permutaciones de los hiperparametros a evaluar 
# La función que llama esto se comentó porque demora mucho crear los datos, se usó y almacenó los datos en un csv

from itertools import product

# Valores posibles para cada hiperparámetro
max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split_values = [2, 5, 10]
min_samples_leaf_values = [1, 2, 4]
criterion_values = ['gini', 'entropy']

combinaciones = list(product(max_depth_values, min_samples_split_values, min_samples_leaf_values, criterion_values))

#for combinacion in combinaciones:
#   print(combinacion)


#%% Función para evaluar todas las combinaciones de hiperparámetros manualmente
# La función que llama esto se comentó porque demora mucho crear los datos, se usó y almacenó los datos en un csv
def evaluar_combinaciones_manual(X, y, combinaciones):
    resultados_completos = [] 
    
    for max_depth, min_samples_split, min_samples_leaf, criterion in combinaciones:
        print(f"\nEvaluando modelo con: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={criterion}")
        resultados = evaluar_arbol(X, y, max_depth, min_samples_split, min_samples_leaf, criterion)
        
        for resultado in resultados:
            resultados_completos.append({
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion,
                'fold': resultado['fold'],
                'precision': resultado['precision'],
                'recall': resultado['recall'],
                'f1': resultado['f1'],
                'confusion_matrix': resultado['confusion_matrix']
            })
    
    resultados_df = pd.DataFrame(resultados_completos)
    return resultados_df


#%%######################

####   La creación de estos datos dura mucho y el resultado 
###    se guardó en resultados_combinaciones.csv

#########################
#%%  Evaluar las combinaciones y guardar los resultados
#resultados = evaluar_combinaciones_manual(X, y, combinaciones)

#print(resultados)

#resultados.to_csv('resultados_combinaciones.csv', index=False)

#%%

# Agrupar y calcular el promedio de precision, recall y f1
resultado_combinacion_grouped = resultado_combinacion.groupby(['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion']).agg({
    'precision': 'mean',
    'recall': 'mean',
    'f1': 'mean'
}).reset_index()

# Obtener la mejor combinación de hiperparámetros
mejor_combinacion = resultado_combinacion_grouped.loc[resultado_combinacion_grouped['f1'].idxmax()]
print("Mejor combinación de hiperparámetros:")
print(mejor_combinacion)

# Filtrar el DataFrame original para obtener la fila completa con la mejor combinación
mejor_filas = resultado_combinacion[
    (resultado_combinacion['max_depth'] == mejor_combinacion['max_depth']) &
    (resultado_combinacion['min_samples_split'] == mejor_combinacion['min_samples_split']) &
    (resultado_combinacion['min_samples_leaf'] == mejor_combinacion['min_samples_leaf']) &
    (resultado_combinacion['criterion'] == mejor_combinacion['criterion'])
]


#%% Graficar la matriz de arriba 

# Seleccionar una matriz de confusión de una de las "folds"
confusion_matrix_str = mejor_filas.iloc[0]['confusion_matrix'] 


# Limpiar la cadena para poder convertirla en una lista de listas
confusion_matrix_str = re.sub(r'\s+', ',', confusion_matrix_str.replace('\n', '').replace('[', '').replace(']', ''))
confusion_matrix_list = [list(map(int, row.split())) for row in confusion_matrix_str.split(',') if row]


confusion_matrix_ = np.array(confusion_matrix_list).reshape(10, 10)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(" Matriz de confusion de con la Mejor Combinación de Hiperparámetros")
plt.show()


#%% Arbol de la mejor combinación 

mejor_modelo = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    criterion='gini'
)

mejor_modelo.fit(X, y)


plt.figure(figsize=(20, 10))
plot_tree(mejor_modelo, filled=True, rounded=True, class_names=True, feature_names=X.columns)
plt.title("Árbol de Decisión con la Mejor Combinación de Hiperparámetros")
plt.show()


#%%######################

#### Ver Held out  

#########################
#%% Uso mi mejor modelo con los datos en held out 

X_held_out = held_out_pixeles.drop(columns=['labels'])
y_held_out = held_out_pixeles['labels']

# Predecir 
y_pred_held_out = mejor_modelo.predict(X_held_out)

# Calcular (accuracy)
exactitud = accuracy_score(y_held_out, y_pred_held_out)
print(f"Exactitud en held-out: {exactitud:.4f}")

# Crear matriz
matriz_confusion_held_out = confusion_matrix(y_held_out, y_pred_held_out)

# Graficar 
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_confusion_held_out, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.title("Matriz de Confusión en el Conjunto Held-out")
plt.show()


