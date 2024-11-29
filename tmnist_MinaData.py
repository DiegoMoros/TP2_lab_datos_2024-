"""
Equipo: MinaData
Integrantes: Diego Moros, Nicole Britos, María Obregón
Descripción del trabajo: 
Este trabajo busca analizar y comprender el dataset TMNIST_Data, con el fin de crear un modelo que nos permita predecir
El codigo esta dividio en 2 modulos, que usamos para poder establer el analisis
Modulos:  
    - Analisis_Exploratorio
        -busqueda
    - Modelos_predictivos
        -clasificacion_binaria
        -clasificacion_multiclase
"""
#%%
from Analisis_Exploratorio import busqueda as bd
from Modelos_predictivos import clasificacion_binaria as cb
from Modelos_predictivos import clasificacion_multiclase as cmc
import pandas as pd
#%%
DATA_PATH = r"Datos\TMNIST_Data.csv"
RESULTADOS = pd.read_csv(r'Datos\resultados_combinaciones.csv')
pixeles = pd.read_csv(DATA_PATH)
report_info = bd.inf_general(pixeles)
# %%
# Parte 1 analisis excploratorio
print("Iniciamos el analisis del conjunto de datos TMNIST")

print("Adjuntamos la información general del dataset, como  sus atributos y su información más relevante")

print("-------Informacion---------")
print(report_info['info'])
print("-------Describe---------")
print(report_info['describe'])
print("-------Instnacias---------")
print(report_info['instancias'])
"""
¿Cuáles parecen ser atributos relevantes para predecir el número al
 que corresponde la imagen? ¿Cuáles no? ¿Creen que se pueden
 descartar atributos?
"""
bd.corr_pixel_for_tag(pixeles)
bd.corr_pixel_for_tag(pixeles, method='spearman')
"""
 ¿Hay números que son más parecidos entre sí? Por ejemplo, ¿Qué es
 más fácil de diferenciar: las imágenes correspondientes al dígito 1 de
 las de el dígito 3, ó las del dígito 3 del dígito 8?
"""
bd.show_pixel_for_tag(pixeles)
bd.show_std_for_tag(pixeles)
bd.comparar_clases_visualmente(pixeles,samples=10)
bd.comparar_clases_visualmente(pixeles,1,4,samples=10)
bd.comparar_clases_visualmente(pixeles,1,3,samples=10)
bd.comparar_clases_visualmente(pixeles,2,5,samples=10)
bd.comparar_clases_metricas(pixeles, digit1=3, digit2=8)
"""
Tomen una de las clases, por ejemplo el dígito 0, ¿Son todas las
 imágenes muy similares entre sí?
"""
bd.mostrar_muestras(pixeles,samples=20)
bd.promedio_desviacion(pixeles)
bd.varianza_pixeles(pixeles)
bd.variancia_intra_clase(pixeles)     
bd.comparar_variancias(pixeles)
bd.calcular_variancia_por_clase(pixeles,0)
bd.comparar_variancia_unica(pixeles,0)
bd.pixeles_relevantes(pixeles, digit=0, top_n=100)
bd.plot_pixel_variability_and_position(pixeles, digit=0, top_n=100)
print("Fin del analisis inicial")
#%%
#Parte 2 classificación_binaria

print("Iniciamos el proceso de modelado y aprendizaje usando clasificacion binaria")
# Definir combinaciones de atributos para las pruebas
combinacion_atributos_3 = [
    ['10', '245', '355'], 
    ['50', '100', '150'], 
    ['400', '650', '234']
]

combinacion_atributos_5 = [
    ['10', '245', '355', '478', '505'],
    ['50', '100', '150', '200', '250'],
    ['400', '450', '500', '550', '600']
]

# Reporte de clasificación binaria
reporte = cb.reporte_clasificacion_binaria(pixeles, combinacion_atributos_3)

print("Datos obtenidos en formato de reporte")

# Imprimir resultados del reporte de clasificación binaria
print("¿Balanceado?:", reporte["balanceado"])
print("Precisión:", reporte["precisions"])
print("Exhaustividad:", reporte["recalls"])
print("Exactitud:", reporte["accuracy_scores"])
print("Matrices de confusión:", reporte["confusion_matrices"])

# Gráfico de exactitud
cb.grafico_exactitud(reporte["accuracy_scores"], combinacion_atributos_3)

# Matrices de confusión
cb.graficar_matrices_confusion(reporte["confusion_matrices"], combinacion_atributos_3)

# Gráfico de precisión y exhaustividad
cb.grafico_precision_exhaustividad(reporte["precisions"], reporte["recalls"], combinacion_atributos_3)

print("Reporte de los atributos usados y los parametros usados")

# Entrenar KNN con subconjuntos de 5 atributos
resultados_subconjuntos = cb.entrenar_knn_con_subconjuntos(pixeles)
for atributos, exactitud, matriz, precision, recall in resultados_subconjuntos:
    print(f"Atributos: {atributos}")
    print(f"Exactitud: {exactitud}")
    print(f"Precisión: {precision}")
    print(f"Exhaustividad: {recall}")
    print(f"Matriz de Confusión:\n{matriz}\n")

# Entrenar KNN con diferentes valores de k
resultados_knn_k = cb.entrenar_knn_con_k_valores(pixeles)
print(resultados_knn_k)

# Graficar heatmap de exactitud
cb.graficar_heatmap_exactitud(resultados_knn_k)

valores_k = [1, 3, 5, 7, 10, 15, 20]
cb.graficar_exactitud_modelos_dataset(pixeles, label_col='labels', valores_k=valores_k)
print("Fin del proceso de modelado y aprendizaje usando clasificacion binaria")

print("Iniciamos el proceso de modelado y aprendizaje usando clasificacion multiclase")
#%%
#Parte 3 clasificacion multiclase
# Dividir los datos en conjuntos de entrenamiento y prueba
distribucion = cmc.dividir_datos(pixeles)
X_train, X_test = distribucion['X'], distribucion['X'].copy()
y_train, y_test = distribucion['Y'], distribucion['Y'].copy()

# Evaluar todas las combinaciones de hiperparámetros posibles
# Debido al costo y consumo de tiempo de la función "evaluar_combinaciones_manual" se guardo su resultado en un csv

# resultados_combinaciones = cmc.evaluar_combinaciones_manual(X_train, y_train)
# resultados_combinaciones.to_csv('Datos\resultados_combinaciones.csv', index=False)

# Identificar la mejor combinación de hiperparámetros
mejor_resultado = cmc.mejor_combinacion_hiperparametros(RESULTADOS)
print("Datos obtenidos")
# Mostrar los resultados de la mejor combinación de hiperparámetros
print(f"Mejor combinación de hiperparámetros:\n{mejor_resultado['mejor_combinacion']}")
print("Mejores filas asociadas con la mejor combinación de hiperparámetros:")
print(mejor_resultado["mejores_filas"])

# Graficar la matriz de confusión para la mejor combinación de hiperparámetros
cmc.grafico_matrix_confusion(mejor_resultado["mejores_filas"])

# Graficar el árbol de decisión
mejor_modelo = cmc.grafico_de_decision(X_train, y_train)

cmc.calcular_accuracy_y_graficar(RESULTADOS)
# Evaluar el modelo en el conjunto de datos held-out (prueba)
exactitud = cmc.grafico_matrix_confusion_held(pixeles, mejor_modelo)
print(f"Exactitud en el conjunto Held-out: {exactitud:.4f}")
# %%
