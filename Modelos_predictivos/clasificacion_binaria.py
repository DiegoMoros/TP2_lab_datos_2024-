import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
    plt.figure(figsize=(10, 6))
    plt.bar([' - '.join(atr) for atr in combinacion_atributos], accuracy_scores, color='skyblue')
    plt.xlabel('Subconjunto de Atributos', fontsize=14)
    plt.ylabel('Exactitud', fontsize=14)
    plt.title('Exactitud de KNN para diferentes subconjuntos de atributos', fontsize=18)
    plt.xticks(rotation=45)
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

def grafico_precision_exhaustividad(precisions, recalls, combinacion_atributos):
    x = np.arange(len(combinacion_atributos))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, precisions, width, label='Precisión')
    rects2 = ax.bar(x + width/2, recalls, width, label='Exhaustividad')
    ax.set_xlabel('Subconjunto de Atributos', fontsize=14)
    ax.set_ylabel('Valores', fontsize=14)
    ax.set_title('Precisión y Exhaustividad de KNN para diferentes subconjuntos de atributos', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([' - '.join(atr) for atr in combinacion_atributos], rotation=45)
    ax.legend()
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

def entrenar_knn_con_k_valores(pixeles_completo):
    pixeles = pixeles_completo.iloc[:, 1:]
    subconjunto = pixeles[pixeles['labels'].isin([0, 1])]
    X = subconjunto.drop(columns=['labels'])
    Y = subconjunto['labels']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    combinacion_atributos = [
        ['10', '245', '355'],
        ['50', '100', '150'],
        ['400', '450', '500']
    ]
    
    valores = [1, 3, 5, 7, 10]
    results = []

    for atributos in combinacion_atributos:
        for k in valores:
            X_train_subconjunto = X_train[atributos]
            X_test_subconjunto = X_test[atributos]
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train_subconjunto, Y_train)
            results.append({'Atributos': ', '.join(atributos), 'k': k, 'Exactitud': accuracy_score(Y_test, model.predict(X_test_subconjunto))})

    return pd.DataFrame(results)

def graficar_heatmap_exactitud(results_df):
    heatmap_data = results_df.pivot(index="Atributos", columns="k", values="Exactitud")
    plt.figure(figsize=(12, 6))
    heatmap = sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Exactitud'})
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Exactitud', fontsize=14)
    plt.title('Exactitud de KNN para distintos atributos y valores de k', fontsize=18)
    plt.xlabel('Número de vecinos (k)', fontsize=14)
    plt.ylabel('Conjunto de atributos', fontsize=14)
    plt.show()

def graficar_exactitud_modelos_dataset(dataset, label_col, valores_k):
    """
    Compara la exactitud de un modelo KNN en datos de entrenamiento y prueba
    para diferentes valores de k y visualiza los resultados con regresiones lineales.

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

    # Ajustar regresión lineal para exactitud de entrenamiento
    reg_train = LinearRegression()
    reg_train.fit(np.array(valores_k).reshape(-1, 1), exactitud_train)
    pred_train = reg_train.predict(np.array(valores_k).reshape(-1, 1))

    # Ajustar regresión lineal para exactitud de prueba
    reg_test = LinearRegression()
    reg_test.fit(np.array(valores_k).reshape(-1, 1), exactitud_test)
    pred_test = reg_test.predict(np.array(valores_k).reshape(-1, 1))

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Regresión lineal (entrenamiento)
    plt.plot(valores_k, pred_train, label='Regresión Lineal (Train)', color='blue', linewidth=2)
    # Regresión lineal (prueba)
    plt.plot(valores_k, pred_test, label='Regresión Lineal (Test)', color='red', linewidth=2)

    # Puntos exactitud (entrenamiento)
    plt.scatter(valores_k, exactitud_train, color='blue', label='Exactitud (Train)', zorder=5)
    # Puntos exactitud (prueba)
    plt.scatter(valores_k, exactitud_test, color='red', label='Exactitud (Test)', zorder=5)

    # Personalización del gráfico
    plt.xlabel('Número de Vecinos (k)', fontsize=14)
    plt.ylabel('Exactitud', fontsize=14)
    plt.title('Exactitud del Modelo KNN para diferentes valores de k', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

