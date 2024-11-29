"""
Análisis que explora y visualiza las características del conjunto de datos TMNIST,
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def inf_general(data):
    """
    Genera un reporte general de los datos.
    
    Parámetros:
    - data (DataFrame): El DataFrame con los datos a analizar.
    
    Retorna:
    - dict: Un diccionario con la información general, estadísticas descriptivas y matriz de correlación.
    """
    file = r"Analisis_Exploratorio\reportes_analisis\reporte_general.txt"
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    
    data_desc = {
        "info": info_str,
        "describe": data.describe(),
        "instancias" : data['labels'].value_counts()
    }
    # with open(file, 'w',encoding="UTF-8") as f:
    #     f.write("Información General del Dataset:\n")
    #     f.write(data_desc["info"])
    #     f.write("\n")
        
    #     f.write("Estadísticas Descriptivas:\n")
    #     f.write(data_desc["describe"].to_string())
    #     f.write("\n")

    #     f.write("instancias de la clase digito:\n")
    #     f.write(data_desc["instancias"].to_string())
    #     f.write("\n")
    return data_desc


def show_fonts_whit_n(data,value_to_graf=15):
    """
    """
    img = np.array(data.iloc[value_to_graf, 2:], dtype=np.float32).reshape((28,28))  # Convertir a float32
    plt.imshow(img, cmap='gray')
    plt.title(f"Ejemplo de imagen con etiqueta {data.iloc[value_to_graf, 1]}")
    plt.show()
    
def show_pixel_for_tag(data):
    """
    """
    promedio_pixeles = []

    # Iterar sobre las clases (0-9)
    for etiqueta in range(10):
        clase = data[data.iloc[:, 1] == etiqueta]
        promedio = clase.iloc[:, 2:].mean(axis=0)
        
        promedio_pixeles.append(promedio.values.reshape(28, 28))
    
    promedio_pixeles = np.array(promedio_pixeles)
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(promedio_pixeles[i], cmap='gray')
        ax.set_title(f'Dígito {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def corr_pixel_for_tag(data, method='pearson'):
    """
    Calcula la correlación entre píxeles y etiquetas utilizando el método especificado.
    
    Parameters:
        - data: DataFrame con las imágenes y etiquetas.
        - method: Método de correlación ('pearson', 'spearman', 'kendall').
    """
    output_csv = f'Analisis_Exploratorio/reportes_analisis/correlacion_pixeles_{method}.csv'
    correlaciones = data.iloc[:, 2:].corrwith(data.iloc[:, 1], method=method)
    correlaciones = correlaciones.abs().sort_values(ascending=False)
    correlaciones.to_csv(output_csv, header=True)
    top_correlacion = correlaciones.head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_correlacion.index, y=top_correlacion.values)
    plt.title(f'Píxeles más relevantes para predecir el número ({method})')
    plt.xlabel('Índice del píxel')
    plt.ylabel(f'Correlación ({method}) con la etiqueta')
    plt.show()

def mostrar_muestras(data, digit=0, samples=10):
    """
    Muestra una selección de imágenes de la clase indicada.
    Parameters:
        - data: El DataFrame con las imágenes y etiquetas.
        - digit: La clase a visualizar (por defecto, el dígito 0).
        - samples: Número de muestras a mostrar.
    """
    fig, axes = plt.subplots(2, samples // 2, figsize=(10, 5))
    subset = data[data.iloc[:, 1] == digit].sample(samples, random_state=0)
    for i, ax in enumerate(axes.flat):
        # Asegúrate de convertir los valores de los píxeles a float
        img = np.array(subset.iloc[i, 2:], dtype=np.float32).reshape((28, 28))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Muestras de imágenes de la clase {digit}')
    plt.show()


def promedio_desviacion(data, digit=0):
    """
    Calcula y muestra el promedio y la desviación estándar de las imágenes de una clase específica.
    Parameters:
        - data: El DataFrame con las imágenes y etiquetas.
        - digit: La clase a analizar (por defecto, el dígito 0).
    """
    subset_pixels = data[data.iloc[:, 1] == digit].iloc[:, 2:]
    mean_img = subset_pixels.mean(axis=0).values.reshape((28, 28))
    std_img = subset_pixels.std(axis=0).values.reshape((28, 28))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mean_img, cmap='gray')
    axes[0].set_title(f'Promedio de imágenes del dígito {digit}')
    axes[1].imshow(std_img, cmap='hot')
    axes[1].set_title(f'Desviación estándar de imágenes del dígito {digit}')
    plt.show()
    
def comparar_clases_metricas(data, digit1=3, digit2=8):
    """
    Calcula y grafica el valor absoluto promedio de la diferencia entre los píxeles promedio de dos clases.
    
    Parameters:
        - data: DataFrame con las imágenes y etiquetas.
        - digit1: Primera clase a comparar.
        - digit2: Segunda clase a comparar.
    """
    promedio1 = data[data.iloc[:, 1] == digit1].iloc[:, 2:].mean(axis=0)
    promedio2 = data[data.iloc[:, 1] == digit2].iloc[:, 2:].mean(axis=0)
    
    diferencia_promedio = np.abs(promedio1 - promedio2).values.reshape(28, 28)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(diferencia_promedio, cmap='hot')
    plt.colorbar(label='Diferencia promedio')
    plt.title(f'Diferencia promedio absoluta de píxeles ({digit1} vs {digit2})')
    plt.show()
    
    diferencia_total = np.mean(np.abs(promedio1 - promedio2))
    print(f"Diferencia promedio absoluta entre los dígitos {digit1} y {digit2}: {diferencia_total}")

def varianza_pixeles(data, digit=0):
    """
    Grafica la desviación estándar de cada píxel para las imágenes de una clase específica.
    Parameters:
        - data: El DataFrame con las imágenes y etiquetas.
        - digit: La clase a analizar (por defecto, el dígito 0).
    """
    subset_pixels = data[data.iloc[:, 1] == digit].iloc[:, 2:]
    std_img = subset_pixels.std(axis=0).values.reshape((28, 28))

    plt.figure(figsize=(10, 6))
    plt.plot(std_img.flatten(), label=f'Desviación estándar de cada píxel para el dígito {digit}')
    plt.xlabel('Índice del píxel')
    plt.ylabel('Desviación estándar')
    plt.title(f'Variabilidad de los píxeles en imágenes del dígito {digit}')
    plt.legend()
    plt.show()

def comparar_clases_visualmente(data, digit1=3, digit2=8, samples=5):
    """
    Muestra una comparación visual entre dos clases de dígitos.
    Parameters:
        - data: El DataFrame con las imágenes y etiquetas.
        - digit1: La primera clase a visualizar.
        - digit2: La segunda clase a visualizar.
        - samples: Número de muestras a mostrar de cada clase.
    """
    fig, axes = plt.subplots(2, samples, figsize=(10, 5))
    
    subset1 = data[data.iloc[:, 1] == digit1].sample(samples, random_state=0)
    subset2 = data[data.iloc[:, 1] == digit2].sample(samples, random_state=0)
    
    for i in range(samples):
        img1 = np.array(subset1.iloc[i, 2:], dtype=np.float32).reshape((28, 28))
        img2 = np.array(subset2.iloc[i, 2:], dtype=np.float32).reshape((28, 28))
        
        axes[0, i].imshow(img1, cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(img2, cmap='gray')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Comparación visual entre las clases {digit1} y {digit2}')
    plt.show()
    
def variancia_intra_clase(data, digit=0):
    """
    Calcula la variancia intra-clase para una clase específica.
    
    Parámetros:
        - data (DataFrame): El DataFrame con las imágenes y etiquetas.
        - digit (int): La clase a analizar (por defecto, el dígito 0).
    
    Retorna:
        - variancia_promedio (float): La variancia promedio de los píxeles.
    """
    subset = data[data.iloc[:, 1] == digit].iloc[:, 2:]
    variancia = subset.var(axis=0).mean()
    return variancia

def comparar_variancias(data, digits=list(range(10))):
    """
    Compara la variancia intra-clase entre múltiples dígitos.
    
    Parámetros:
        - data (DataFrame): El DataFrame con las imágenes y etiquetas.
        - digits (list): Lista de dígitos a comparar (por defecto, 0-9).
    """
    variancias = {}
    for digit in digits:
        variancias[digit] = variancia_intra_clase(data, digit)
    
    # Convertir a DataFrame para una mejor visualización
    variancias_df = pd.DataFrame(list(variancias.items()), columns=['Dígito', 'Variancia Promedio'])
    variancias_df = variancias_df.sort_values(by='Variancia Promedio', ascending=False)
    
    # Graficar las variancias
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Dígito', y='Variancia Promedio', data=variancias_df, palette='viridis')
    plt.title('Comparación de la Variancia Intra-clase entre Dígitos')
    plt.xlabel('Dígito')
    plt.ylabel('Variancia Promedio de los Píxeles')
    plt.show()
    
    return variancias_df
def comparar_variancia_unica(data, digit):
    """
    Calcula y muestra la variancia intra-clase para un único dígito.

    Parámetros:
        - data (DataFrame): El DataFrame con las imágenes y etiquetas.
        - digit (int): El dígito específico para el cual calcular la variancia.
    """
    # Calcular la variancia intra-clase para el dígito proporcionado
    variancia = variancia_intra_clase(data, digit)
    
    # Crear un DataFrame para mostrar la variancia
    variancia_df = pd.DataFrame([(digit, variancia)], columns=['Dígito', 'Variancia Promedio'])

    # Graficar la variancia
    plt.figure(figsize=(6, 6))
    sns.barplot(x='Dígito', y='Variancia Promedio', data=variancia_df, palette='viridis')
    plt.title(f'Variancia Intra-clase para el Dígito {digit}')
    plt.xlabel('Dígito')
    plt.ylabel('Variancia Promedio de los Píxeles')
    plt.show()
    
    return variancia_df
def calcular_variancia_por_clase(data, digit):
    """
    Calcula la variancia promedio de los píxeles para las imágenes de una clase específica.
    Parameters:
        - data: El DataFrame con las imágenes y etiquetas.
        - digit: El dígito para el cual se calculará la variancia.
    Returns:
        - varianza_promedio: Variancia promedio de los píxeles en las imágenes de la clase especificada.
    """
    # Filtrar las imágenes correspondientes a la clase digit
    subset = data[data.iloc[:, 1] == digit].iloc[:, 2:]
    
    # Calcular la variancia de cada píxel en el subset
    variancias = subset.var(axis=0)
    
    # Calcular la variancia promedio
    varianza_promedio = variancias.mean()
    print(f"Variancia promedio de los píxeles para el dígito {digit}: {varianza_promedio}")
    return varianza_promedio

def pixeles_relevantes(data, digit=0, top_n=10):
    """
    Identifica y visualiza los píxeles más relevantes para una clase específica.
    
    Parameters:
        - data: DataFrame con las imágenes y etiquetas.
        - digit: La clase a analizar.
        - top_n: Número de píxeles más relevantes a mostrar.
    """
    subset_pixels = data[data.iloc[:, 1] == digit].iloc[:, 2:]
    correlaciones = subset_pixels.corrwith(data.iloc[:, 1], method='spearman').abs()
    top_pixels = correlaciones.sort_values(ascending=False).head(top_n)

    print("Píxeles más relevantes y sus posiciones:")
    for pixel, value in top_pixels.items():
        fila = (int(pixel) - 2) // 28
        columna = (int(pixel) - 2) % 28
        print(f"Píxel {pixel}: Correlación={value:.4f}, Posición (fila, columna)=({fila}, {columna})")

    # Mostrar los píxeles en su posición
    relevancia_imagen = np.zeros((28, 28))
    for pixel in top_pixels.index:
        fila = (int(pixel) - 2) // 28
        columna = (int(pixel) - 2) % 28
        relevancia_imagen[fila, columna] = top_pixels[pixel]

    plt.figure(figsize=(10, 6))
    plt.imshow(relevancia_imagen, cmap='hot')
    plt.colorbar(label='Relevancia')
    plt.title(f'Píxeles más relevantes para el dígito {digit}')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_pixel_variability_and_position(data, digit=0, top_n=10):
    """
    Muestra la variabilidad de los píxeles más relevantes y sus posiciones (centrales o de borde) para una clase específica.
    
    Parameters:
        - data: DataFrame con las imágenes y etiquetas.
        - digit: La clase a analizar.
        - top_n: Número de píxeles más relevantes a mostrar.
    """
    # Obtener los píxeles más relevantes para el dígito seleccionado
    subset_pixels = data[data.iloc[:, 1] == digit].iloc[:, 2:]
    correlaciones = subset_pixels.corrwith(data.iloc[:, 1], method='spearman').abs()
    top_pixels = correlaciones.sort_values(ascending=False).head(top_n)

    # Inicializamos una matriz vacía de 28x28 para resaltar la relevancia
    relevancia_imagen = np.zeros((28, 28))

    # Guardamos la posición de los píxeles relevantes (y si están en el centro o bordes)
    posiciones_relevantes = []

    for pixel, value in top_pixels.items():
        fila = (int(pixel) - 2) // 28
        columna = (int(pixel) - 2) % 28

        # Asignamos el valor de relevancia en la posición correspondiente
        relevancia_imagen[fila, columna] = value

        # Clasificamos si el píxel está en el centro o en los bordes
        if 9 <= fila <= 19 and 9 <= columna <= 19:
            posiciones_relevantes.append((fila, columna, "Centro"))
        else:
            posiciones_relevantes.append((fila, columna, "Borde"))

    # Visualizar la imagen de relevancia
    plt.figure(figsize=(10, 6))
    plt.imshow(relevancia_imagen, cmap='hot')
    plt.colorbar(label='Relevancia')
    plt.title(f'Píxeles más relevantes para el dígito {digit} (Centro vs Borde)')
    plt.show()

    # Imprimir las posiciones de los píxeles relevantes
    print(f"Posiciones de los {top_n} píxeles más relevantes:")
    for fila, columna, posicion in posiciones_relevantes:
        print(f"Píxel en fila {fila}, columna {columna} - Posición: {posicion}")

def show_std_for_tag(data):
    """
    Calcula y muestra la desviación estándar de los píxeles para cada clase (dígitos 0-9).
    
    Parameters:
    - data: DataFrame que contiene los datos, donde la columna 1 es la etiqueta (dígito) y
      las columnas 2 en adelante contienen los valores de los píxeles.
    """
    std_pixeles = []

    # Iterar sobre las clases (0-9)
    for etiqueta in range(10):
        clase = data[data.iloc[:, 1] == etiqueta]
        std = clase.iloc[:, 2:].std(axis=0)  # Cambiar mean por std para calcular desviación estándar
        
        std_pixeles.append(std.values.reshape(28, 28))
    
    std_pixeles = np.array(std_pixeles)
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(std_pixeles[i], cmap='hot')  # Usar un mapa de calor para visualizar mejor las variaciones
        ax.set_title(f'Dígito {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()