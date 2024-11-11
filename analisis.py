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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier , plot_tree

#%% Cargar los datos
pixeles_completo = pd.read_csv('C:/Users/mgo20/OneDrive/Desktop/Data/plab/TP2_lab_datos_2024-/Datos/TMNIST_Data.csv')

#%% Quitar la fuente 
pixeles = pixeles_completo.iloc[:, 1:]

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
print(pixeles['labels'].value_counts())  # Los datos están balanceados

#%% Dividir los datos en conjuntos de desarrollo (dev) y de prueba (held-out)
dev_pixeles, held_out_pixeles = train_test_split(pixeles, test_size=0.2, stratify=pixeles['labels'], random_state=42)

print("Distribución en dev:")
print(dev_pixeles['labels'].value_counts())
print("\nDistribución en held-out:")
print(held_out_pixeles['labels'].value_counts())

X = dev_pixeles.drop(columns=['labels'])
y = dev_pixeles['labels']

#%% Definir función para calcular métricas de evaluación
def calcular_metricas(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))  # Verdaderos positivos
    fp = np.sum((y_pred == 1) & (y_true == 0))  # Falsos positivos
    fn = np.sum((y_pred == 0) & (y_true == 1))  # Falsos negativos
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

#%% Función para evaluar el modelo de árbol de decisión con combinaciones de hiperparámetros
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

#%%


from itertools import product

# Valores posibles para cada hiperparámetro
max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split_values = [2, 5, 10]
min_samples_leaf_values = [1, 2, 4]
criterion_values = ['gini', 'entropy']

# Generar todas las combinaciones de los parámetros
combinaciones = list(product(max_depth_values, min_samples_split_values, min_samples_leaf_values, criterion_values))

# Imprimir las combinaciones generadas
for combinacion in combinaciones:
    print(combinacion)


#%% Función para evaluar todas las combinaciones de hiperparámetros manualmente
def evaluar_combinaciones_manual(X, y, combinaciones):
    resultados_completos = []  # Lista para guardar los resultados de todas las combinaciones
    
    for max_depth, min_samples_split, min_samples_leaf, criterion in combinaciones:
        print(f"\nEvaluando modelo con: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={criterion}")
        resultados = evaluar_arbol(X, y, max_depth, min_samples_split, min_samples_leaf, criterion)
        
        # Guardar los resultados para cada combinación
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
    
    # Convertir los resultados a un DataFrame
    resultados_df = pd.DataFrame(resultados_completos)
    return resultados_df


#%%######################

####   La creación de estos datos dura mucho y el resultado 
###    se guardó en resultados_combinaciones.csv

#########################
#%% Paso 3: Evaluar las combinaciones y guardar los resultados
#resultados = evaluar_combinaciones_manual(X, y, combinaciones)

#%% Paso 4: Mostrar los resultados
#print(resultados)

#%% Paso 5: Guardar los resultados en un archivo CSV si se desea
#resultados.to_csv('resultados_combinaciones.csv', index=False)

#%%

# Cargo el DataFrame de las matrices de cada combinación 
resultado_combinacion = pd.read_csv('C:/Users/mgo20/OneDrive/Desktop/Data/plab/TP2_lab_datos_2024-/Datos/resultados_combinaciones.csv')

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

#%%%
# Filtrar el DataFrame original para obtener la fila completa con la mejor combinación
mejor_filas = resultado_combinacion[
    (resultado_combinacion['max_depth'] == mejor_combinacion['max_depth']) &
    (resultado_combinacion['min_samples_split'] == mejor_combinacion['min_samples_split']) &
    (resultado_combinacion['min_samples_leaf'] == mejor_combinacion['min_samples_leaf']) &
    (resultado_combinacion['criterion'] == mejor_combinacion['criterion'])
]

# Crear un heatmap de los valores F1 por max_depth y min_samples_split para el mejor valor de min_samples_leaf y criterion
plt.figure(figsize=(12, 8))
resultado_combinacion_heatmap = resultado_combinacion_grouped[
    (resultado_combinacion_grouped['min_samples_leaf'] == mejor_combinacion['min_samples_leaf']) &
    (resultado_combinacion_grouped['criterion'] == mejor_combinacion['criterion'])
]

# Eliminar posibles duplicados promediando valores
resultado_combinacion_heatmap = resultado_combinacion_heatmap.groupby(['max_depth', 'min_samples_split']).agg({'f1': 'mean'}).reset_index()

# Crear la tabla pivote para el heatmap
pivot_table = resultado_combinacion_heatmap.pivot(index="max_depth", columns="min_samples_split", values="f1")

sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title(f"Heatmap del F1 promedio por max_depth y min_samples_split\n(mejor min_samples_leaf = {int(mejor_combinacion['min_samples_leaf'])}, criterion = {mejor_combinacion['criterion']})")
plt.xlabel("min_samples_split")
plt.ylabel("max_depth")
plt.show()


#%%

# Seleccionar una matriz de confusión de una de las "folds"
confusion_matrix_str = mejor_filas.iloc[0]['confusion_matrix']  # Extrae la primera matriz de confusión para esta configuración


# Limpiar la cadena para poder convertirla en una lista de listas
confusion_matrix_str = re.sub(r'\s+', ',', confusion_matrix_str.replace('\n', '').replace('[', '').replace(']', ''))
confusion_matrix_list = [list(map(int, row.split())) for row in confusion_matrix_str.split(',') if row]

# Convertir la lista de listas en un array de Numpy
confusion_matrix = np.array(confusion_matrix_list).reshape(10, 10)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
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

#%%



