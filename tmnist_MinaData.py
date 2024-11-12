"""
Equipo: MinaData
Integrantes: Diego Moros, Nicole Britos, María Obregón
Descripción del trabajo: 
Modulos: 
"""

from Analisis_Exploratorio import busqueda as bd
import pandas as pd
from numpy import random
DATA_PATH = r"Datos\TMNIST_Data.csv"
data = pd.read_csv(DATA_PATH)
report_info = bd.inf_general(data)
# #Parte 1 analisis excploratorio
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
bd.corr_pixel_for_tag(data)
"""
 ¿Hay números que son más parecidos entre sí? Por ejemplo, ¿Qué es
 más fácil de diferenciar: las imágenes correspondientes al dígito 1 de
 las de el dígito 3, ó las del dígito 3 del dígito 8?
"""
bd.show_pixel_for_tag(data)
bd.comparar_clases_visualmente(data,samples=10)
bd.comparar_clases_visualmente(data,1,4,samples=10)
bd.comparar_clases_visualmente(data,1,3,samples=10)
bd.comparar_clases_visualmente(data,2,5,samples=10)
"""
Tomen una de las clases, por ejemplo el dígito 0, ¿Son todas las
 imágenes muy similares entre sí?
"""
bd.mostrar_muestras(data,samples=20)
bd.promedio_desviacion(data)
bd.varianza_pixeles(data)
bd.variancia_intra_clase(data)
bd.comparar_variancias(data)
bd.calcular_variancia_por_clase(data,0)

""" 
Este dataset está compuesto por imágenes, esto plantea una
 diferencia frente a los datos que utilizamos en las clases (por ejemplo,
 el dataset de Titanic). ¿Creen que esto complica la exploración de los
 datos?
"""  