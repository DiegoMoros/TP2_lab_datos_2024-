import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
from sklearn.linear_model import LinearRegression

def reporte_clasificacion_binaria(pixeles_completo, combinacion_atributos):
    report_dict = {}
    pixeles = pixeles_completo.iloc[:, 1:]
    subconjunto = pixeles[pixeles['labels'].isin([0, 1])]
    contador_de_clase = subconjunto['labels'].value_counts()
    report_dict["balanceado"] = (contador_de_clase[0] == contador_de_clase[1])
    
    X = subconjunto.drop(columns=['labels'])
    Y = subconjunto['labels']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    report_dict["precisions"] = []
    report_dict["recalls"] = []
    report_dict["accuracy_scores"] = []
    report_dict["confusion_matrices"] = []
    
    for atributo in combinacion_atributos:
        X_train_subconjunto = X_train[atributo]
        X_test_subconjunto = X_test[atributo]

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train_subconjunto, Y_train)
        Y_pred = model.predict(X_test_subconjunto)

        report_dict["accuracy_scores"].append(accuracy_score(Y_test, Y_pred))
        report_dict["precisions"].append(precision_score(Y_test, Y_pred, average='binary'))
        report_dict["recalls"].append(recall_score(Y_test, Y_pred, average='binary'))
        report_dict["confusion_matrices"].append(confusion_matrix(Y_test, Y_pred))
        
    return report_dict



def grafico_exactitud(accuracy_scores, combinacion_atributos):
    plt.figure(figsize=(6, 4))  # Ajusta el tamaño del gráfico
    
    # Gráfico de puntos (scatter plot)
    plt.scatter([' - '.join(atr) for atr in combinacion_atributos], accuracy_scores, color='skyblue', s=100, label='Exactitud')
    
    # Personalización del gráfico
    plt.xlabel('Subconjunto de Atributos', fontsize=10, labelpad=5)  # Ajusta el padding
    plt.ylabel('Exactitud', fontsize=10)
    plt.title('Exactitud de KNN para diferentes subconjuntos de atributos', fontsize=12)
    plt.xticks(rotation=3, fontsize=10)  # Reduce la rotación y el tamaño de las etiquetas
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()  # Asegura que las etiquetas no se recorten
    plt.show()



def graficar_matrices_confusion(confusion_matrices, combinacion_atributos):
    fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(18, 5))
    for i, (atr, cm) in enumerate(zip(combinacion_atributos, confusion_matrices)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"Matriz de Confusión para atributos {' - '.join(atr)}", fontsize=14)
        axes[i].set_xlabel('Predicción', fontsize=14)
        axes[i].set_ylabel('Real', fontsize=14)

    plt.tight_layout()
    plt.show()


def entrenar_knn_con_subconjuntos(pixeles_completo):
    pixeles = pixeles_completo.iloc[:, 1:]
    subconjunto = pixeles[pixeles['labels'].isin([0, 1])]
    X = subconjunto.drop(columns=['labels'])
    Y = subconjunto['labels']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    combinacion_atributos = [
        ['10', '245', '355', '478', '505'],
        ['50', '100', '150', '200', '250'],
        ['400', '450', '500', '550', '600']
    ]
    
    results = []
    for atributo in combinacion_atributos:
        X_train_subconjunto = X_train[atributo]
        X_test_subconjunto = X_test[atributo]
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train_subconjunto, Y_train)
        Y_pred = model.predict(X_test_subconjunto)
        results.append((atributo, accuracy_score(Y_test, Y_pred), confusion_matrix(Y_test, Y_pred),
                        precision_score(Y_test, Y_pred, average='binary'), recall_score(Y_test, Y_pred, average='binary')))
        
    return results



def graficar_exactitud_modelos_dataset(dataset, label_col, valores_k):
    """
    Compara la exactitud de un modelo KNN en datos de entrenamiento y prueba
    para diferentes valores de k y visualiza los resultados conectando los puntos.

    Parámetros:
    - dataset: DataFrame o matriz que contiene los datos completos.
    - label_col: Índice o nombre de la columna que contiene las etiquetas (clase o dígito).
    - valores_k: Lista de valores de k para KNN.
    """
    # Separar características (X) y etiquetas (Y)
    # Seleccionar solo las columnas numéricas
    X = dataset.select_dtypes(include=[np.number]).drop(columns=[label_col]).values
    Y = dataset[label_col].values  # Columna con las etiquetas
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # Listas para almacenar exactitudes
    exactitud_train = []
    exactitud_test = []

    # Calcular exactitudes para diferentes valores de k
    for k in valores_k:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)
        
        # Calcular exactitud en datos de entrenamiento
        Y_train_pred = model.predict(X_train)
        exactitud_train.append(accuracy_score(Y_train, Y_train_pred))
        
        # Calcular exactitud en datos de prueba
        Y_test_pred = model.predict(X_test)
        exactitud_test.append(accuracy_score(Y_test, Y_test_pred))

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Conectar puntos exactitud (entrenamiento)
    plt.plot(valores_k, exactitud_train, label='Exactitud (Train)', color='blue', linewidth=2)
    # Conectar puntos exactitud (prueba)
    plt.plot(valores_k, exactitud_test, label='Exactitud (Test)', color='red', linewidth=2)

    # Personalización del gráfico
    plt.xlabel('Número de Vecinos (k)', fontsize=14)
    plt.ylabel('Exactitud', fontsize=14)
    plt.title('Exactitud del Modelo KNN para diferentes valores de k', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
