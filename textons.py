import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat

# TODO Copiar y pegar estas funciones en el script principal (main_Codigo1_Codigo2.py)
# TODO Cambiar el nombre de las funciones para incluir sus códigos de estudiante

def calculateFilterResponse_Codigo1_Codigo2(img_gray, filters):
    
    # TODO Inicializar arreglo de tamaño (MxN) x número de filtros, llamado 'resp'
    # TODO Realizar un (1) ciclo que recorra los filtros
    # TODO En cada iteración: 
    #           - Realizar cross-correlación entre la imagen y el filtro. Para ello, utilizar
    #             correlate2d() y los parámetros que considere pertinentes para no perder el
    #             tamaño original de la imagen.
    #           - Convertir el resultado a un vector y almacenarlo en la posición correspondiente 
    #             del arreglo inicial. 
    return resp

def calculateTextonDictionary_Codigo1_Codigo2(images_train, filters, parameters):

    # TODO Inicializar arreglo de respuestas de tamaño [(MxN) x número de imágenes] x número de filtros
    # TODO Realizar un (1) ciclo que recorra todas las imágenes de entrenamiento
    # TODO En cada iteración:
    #           - Calcular la respuesta de la imagen al banco de filtros (función anterior)
    #           - Almacenar la matriz resultante en el arreglo de respuestas (tenga en cuenta
    #             la posición de los pixeles de cada imagen dentro del arreglo de respuestas)

    # TODO Establecer semilla
    # TODO Declarar el modelo de KMeans
    # TODO Ajustar el modelo inicializado al arreglo de resultados del punto anterior
    # TODO Obtener las coordenadas de los centroides en una variable y almacenarlas 
    #       en un diccionario, bajo la llave 'centroids'
    # TODO Almacenar el diccionario anterior como un archivo .mat, bajo el nombre 
    #       'dictname' (parámetro de entrada)

# TODO Borrar los comentarios marcados con un TODO.