import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier , plot_tree
from itertools import product
import ast

def dividir_datos(pixeles_completo):
    pixeles = pixeles_completo.iloc[:, 1:]
    #Dividir los datos en conjuntos de desarrollo (dev) y de prueba (held-out)
    dev_pixeles, held_out_pixeles = train_test_split(pixeles, test_size=0.2, stratify=pixeles['labels'], random_state=42)
    distribucion = {
        "Dev" : dev_pixeles['labels'].value_counts(),
        "Held-out" : held_out_pixeles['labels'].value_counts(),
        "X" : dev_pixeles.drop(columns=['labels']),
        "Y" : dev_pixeles['labels']
    }
    return distribucion

def calcular_metricas(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))  # Verdaderos positivos
    fp = np.sum((y_pred == 1) & (y_true == 0))  # Falsos positivos
    fn = np.sum((y_pred == 0) & (y_true == 1))  # Falsos negativos
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

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

def generar_valores():
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_split_values = [2, 5, 10]
    min_samples_leaf_values = [1, 2, 4]
    criterion_values = ['gini', 'entropy']

    combinaciones = list(product(max_depth_values, min_samples_split_values, min_samples_leaf_values, criterion_values))
    return combinaciones

def evaluar_combinaciones_manual(x, y):
    resultados_completos = [] 
    combinaciones = generar_valores()
    reporte_str = {}
    for max_depth, min_samples_split, min_samples_leaf, criterion in combinaciones:
        reporte_str["Valores usado"] = f"Evaluando modelo con: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={criterion}"
        resultados = evaluar_arbol(x, y, max_depth, min_samples_split, min_samples_leaf, criterion)
        
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

def mejor_combinacion_hiperparametros(resultado_combinacion):
    resultado_combinacion_grouped = resultado_combinacion.groupby(['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion']).agg({
    'precision': 'mean',
    'recall': 'mean',
    'f1': 'mean'
    }).reset_index()
    mejor_combinacion = resultado_combinacion_grouped.loc[resultado_combinacion_grouped['f1'].idxmax()]
    mejor_filas = resultado_combinacion[
        (resultado_combinacion['max_depth'] == mejor_combinacion['max_depth']) &
        (resultado_combinacion['min_samples_split'] == mejor_combinacion['min_samples_split']) &
        (resultado_combinacion['min_samples_leaf'] == mejor_combinacion['min_samples_leaf']) &
        (resultado_combinacion['criterion'] == mejor_combinacion['criterion'])
    ]
    resultados ={
        "mejor_combinacion" : mejor_combinacion,
        "mejores_filas" : mejor_filas
    }
    return resultados

def grafico_matrix_confusion(mejor_filas):
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


def grafico_de_decision(x,y):
    mejor_modelo = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        criterion='gini'
    )

    mejor_modelo.fit(x, y)


    plt.figure(figsize=(20, 10))
    plot_tree(mejor_modelo, filled=True, rounded=True, class_names=True, feature_names=x.columns)
    plt.title("Árbol de Decisión con la Mejor Combinación de Hiperparámetros")
    plt.show()
    return mejor_modelo
    
def grafico_matrix_confusion_held(pixeles_completo,mejor_modelo):
    pixeles = pixeles_completo.iloc[:, 1:]
    #Dividir los datos en conjuntos de desarrollo (dev) y de prueba (held-out)
    dev_pixeles, held_out_pixeles = train_test_split(pixeles, test_size=0.2, stratify=pixeles['labels'], random_state=42)
    X_held_out = held_out_pixeles.drop(columns=['labels'])
    y_held_out = held_out_pixeles['labels']

    # Predecir 
    y_pred_held_out = mejor_modelo.predict(X_held_out)

    # Calcular (accuracy)
    exactitud = accuracy_score(y_held_out, y_pred_held_out)

    # Crear matriz
    matriz_confusion_held_out = confusion_matrix(y_held_out, y_pred_held_out)

    # Graficar 
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion_held_out, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.title("Matriz de Confusión en el Conjunto Held-out")
    plt.show()
    return exactitud

def calcular_accuracy_y_graficar(df):
    """
    Calcula la exactitud desde una columna de matrices de confusión en un DataFrame,
    guarda los resultados en un archivo de salida y genera un gráfico de tendencia entre 
    la máxima profundidad (max_depth), el criterio (criterion) y la exactitud (accuracy).

    Parámetros:
    - df: DataFrame que contiene la columna 'confusion_matrix' con las matrices de confusión en formato string.
    """
    
    def calcular_accuracy(conf_matrix_str):
        """
        Calcula la exactitud desde una matriz de confusión en formato de string.
        """
        # Limpiar el string de la matriz de confusión
        conf_matrix_str = re.sub(r',+', ',', conf_matrix_str)  # Eliminar comas extra
        conf_matrix_str = conf_matrix_str.strip()  # Eliminar espacios innecesarios

        # Reemplazar las celdas vacías representadas por ",," con ceros
        conf_matrix_str = re.sub(r',,*', ',0,', conf_matrix_str)

        # Asegurarse de que no haya comas al principio o al final
        conf_matrix_str = conf_matrix_str.strip(',')

        # Reemplazar los espacios entre números por comas (en caso de que haya espacios)
        conf_matrix_str = conf_matrix_str.replace(' ', ',')

        conf_matrix_str = re.sub(r',,*', ',', conf_matrix_str)
        # Intentar evaluar el string a una matriz de confusión

        # Usamos ast.literal_eval para convertir la cadena en una lista
        conf_matrix_str = conf_matrix_str.replace('[,', '[').replace(',]', ']')  # Corregir corchetes mal formateados
        conf_matrix = np.array(ast.literal_eval(conf_matrix_str))

        # Calcular la exactitud
        correct_predictions = np.trace(conf_matrix)  # Suma de la diagonal
        total_predictions = np.sum(conf_matrix)  # Suma de todos los valores
        return correct_predictions / total_predictions

    # Calcular la exactitud para cada fila de la columna 'confusion_matrix'
    df['accuracy'] = df['confusion_matrix'].apply(calcular_accuracy)

    # Ahora obtenemos la columna de 'max_depth' y 'criterion' y graficamos la comparación
    plt.figure(figsize=(12, 6))

    # Usamos un gráfico de dispersión para comparar max_depth con la exactitud
    sns.scatterplot(x='max_depth', y='accuracy', hue='criterion', data=df, s=100, alpha=0.7)

    # Añadir líneas de tendencia para cada valor de criterion
    for criterion_value in df['criterion'].unique():
        subset = df[df['criterion'] == criterion_value]
        sns.lineplot(x='max_depth', y='accuracy', data=subset, label=f'Criterion: {criterion_value}', marker='o')

    # Agregar etiquetas y título
    plt.xlabel('Max Depth', fontsize=14)
    plt.ylabel('Exactitud', fontsize=14)
    plt.title('Comparación entre Max Depth, Criterion y Exactitud', fontsize=16)

    # Mostrar la leyenda y la cuadrícula
    plt.legend()
    plt.grid(True)

    # Ajustar el layout y mostrar el gráfico
    plt.tight_layout()
    plt.show()
