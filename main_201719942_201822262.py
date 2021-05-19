#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto4 Entrega1
##Se importan librerías que se utilizarán para el desarrollo del laboratorio
import os
import glob
import pickle
from statistics import mode
import numpy as np
import sklearn.cluster as skclust
import sklearn.metrics as sk
import tqdm as tqdm
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from skimage import img_as_float
from data_mp4.functions import JointColorHistogram, CatColorHistogram
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat
from scipy.spatial import distance
from sklearn import svm
from tqdm import tqdm

def calculate_descriptors(data, parameters, calculate_dict):
    filtros = loadmat('filterbank.mat')
    dataGris = list(map(parameters['transform_color_function'], data))
    if calculate_dict:
        calculateTextonDictionary_201719942_201822262(data, filtros, parameters)
    dic = loadmat(parameters['dict_name'])
    centroides = dic['centroids']
    # descriptor_matrix = np.zeros(len(data), parameters['k'])
    descriptor_matrix = list(map(CalculateTextonHistogram_201719942_201822262, dataGris, centroides))
    print(descriptor_matrix.shape())
    assert len(descriptor_matrix)*len(descriptor_matrix[0]) == (filtros['filterbank'].shape[2])*len(data)*len(data[0][0])*len(data[0][0][0]), 'El tamaño del descriptor no es el adecuado' # verificación
    return descriptor_matrix

diccionario = {} # variable global para guardar cluster asignado para cada una de las clases trabajadas
def train(parameters, action):
    data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')  # carga imágenes train
    rutas = glob.glob(data_train)
    images_train = list(map(io.imread, rutas))
    nombres = []
    global diccionario # variable global para conteo de asignación de clusters para cada clase
    for i in rutas:
        ruta = i.split(i[8]) # extracción de anotación            #print( ruta[-1])
        nombres.append(ruta[-1].split('_')[0])
    if action == 'save':
        descriptors = calculate_descriptors(images_train, parameters, True)
        # TODO Guardar matriz de descriptores con el nombre parameters['train_descriptor_name']
        np.save(parameters['train_descriptor_name'], descriptors)         #print(descriptors)
    else:
        y = 0
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['train_descriptor_name']
        #  np.load(parameters['train_descriptor_name'])
    # TODO Definir una semilla y utilice la misma para todos los experimentos de la entrega.
    kernel = parameters['kernel']
    # TODO Inicializar y entrenar el modelo con los descriptores.
    entrenamiento_SVM = svm.SVC(kernel=kernel).fit(descriptors, nombres)
    #etiquetas = entrenamiento.labels_
    #plt.figure()  # se plotean las imágenes resultantes
    '''for i in range(len(etiquetas)):
        if nombres[i] not in diccionario: #conteo de asignación de clusters para cada una de las etiquetas
            #diccionario[nombres[i]] = [etiquetas[i]]
            diccionario[nombres[i]] = {etiquetas[i]: 1}
        else:
            if etiquetas[i] not in diccionario[nombres[i]]:
                diccionario[nombres[i]][etiquetas[i]] = 1
            else:
                diccionario[nombres[i]][etiquetas[i]] += 1
            #diccionario[nombres[i]].append(etiquetas[i])       #diccionario[data_train[i]] = etiquetas[i]
        plt.subplot(8,6,i+1)
        plt.title(f"{nombres[i]}:\nCluster{etiquetas[i]}")
        plt.axis("off")
        plt.imshow(images_train[i])
        plt.tight_layout()
    #print(diccionario)
    plt.show()'''
    # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
    pickle.dump(entrenamiento_SVM, open(parameters['name_model'],'wb'))

def validate(parameters, action):
    data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
    rutas = glob.glob(data_val)
    images_val = list(map(io.imread, rutas))
    nombres = []
    for i in rutas:
        ruta = i.split(i[8])
        nombres.append(ruta[-1].split('_')[0])
    if action == 'load':
        y = 0
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['val_descriptor_name']
        #descriptors = np.load(parameters['val_descriptor_name'])
    else:
        descriptors = calculate_descriptors(images_val, parameters, True)
        if action == 'save':
            # TODO Guardar matriz de descriptores con el nombre parameters['val_descriptor_name']
            np.save(parameters['val_descriptor_name'], descriptors)
    # TODO Cargar el modelo de parameters['name_model']
    modelo = pickle.load(open(parameters['name_model'], 'rb'))        #print(type(modelo))
    # TODO Obtener las predicciones para los descriptores de las imágenes de validación
    predicciones = modelo.predict(descriptors)
    '''anotaciones = []
    for i in nombres:  # Diccionario con el cual se determinó qué clase correspondía a que anotación: {'buildings': {2: 2, 1: 2, 4: 2, 0: 1, 5: 1}, 'forest': {3: 5, 4: 3}, 'glacier': {1: 1, 5: 3, 0: 2, 4: 1, 3: 1}, 'mountains': {4: 5, 3: 2, 1: 1}, 'sea': {4: 4, 1: 3, 2: 1}, 'street': {1: 4, 3: 1, 0: 2, 2: 1}}
        if i == "glacier":
            anotaciones.append(5)#                                                   anotaciones.append(mode(diccionario['glacier']))
        elif i == "buildings":
            anotaciones.append(0)#                                                 anotaciones.append(mode(diccionario['buildings']))
        elif i == "forest":
            anotaciones.append(3)#                                                anotaciones.append(mode(diccionario['forest']))
        elif i == "mountains":
            anotaciones.append(4)#                                             anotaciones.append(mode(diccionario['mountains']))
        elif i == "sea":
            anotaciones.append(2)#                                             anotaciones.append(mode(diccionario['sea']))
        else:
            anotaciones.append(1)#                                              anotaciones.append(mode(diccionario['street']))'''
    # TODO Obtener las métricas de evaluación
    conf_mat = sk.confusion_matrix(nombres, predicciones)
    precision = sk.precision_score(nombres, predicciones, average="macro")
    recall = sk.recall_score(nombres, predicciones, average="macro")
    f_score = sk.f1_score(nombres, predicciones, average="macro")                                                                                                                                #print((2*precision*recall)/(precision+recall))
    return conf_mat, precision, recall, f_score
# TODO Copiar y pegar estas funciones en el script principal (main_Codigo1_Codigo2.py)
# TODO Cambiar el nombre de las funciones para incluir sus códigos de estudiante

def calculateFilterResponse_201719942_201822262(img_gray, filters):
    # TODO Inicializar arreglo de tamaño (MxN) x número de filtros, llamado 'resp'
    #print((len(img_gray)*len(img_gray[0])))
    cant = filters['filterbank'].shape[2]
    #print(cant)
    resp = np.zeros((len(img_gray)*len(img_gray[0]), cant))
    # TODO Realizar un (1) ciclo que recorra los filtros
    for i in tqdm(range(filters['filterbank'].shape[2])):
        #print(filters['filterbank'])
        correlacion = correlate2d(img_gray, filters['filterbank'][:, :, i], boundary="symm", mode='same')
        #print(filters['filterbank'][:,:,i].shape)
        #print(len(correlacion) * len(correlacion[0]))
        correlacion = np.transpose(np.array([correlacion.flatten()]))
        #print(correlacion)
        #print(len(correlacion))
        resp[:, i] = (correlacion)[:, 0]
        #print(resp)
    # TODO En cada iteración:
    #           - Realizar cross-correlación entre la imagen y el filtro. Para ello, utilizar
    #             correlate2d() y los parámetros que considere pertinentes para no perder el
    #             tamaño original de la imagen.
    #           - Convertir el resultado a un vector y almacenarlo en la posición correspondiente
    #             del arreglo inicial.
    return resp


def calculateTextonDictionary_201719942_201822262(images_train, filters, parameters):
    cant = filters['filterbank'].shape[2]
    # TODO Inicializar arreglo de respuestas de tamaño [(MxN) x número de imágenes] x número de filtros
    M_N = len(images_train[0]) * len(images_train[0][0])
    resp = np.zeros(((M_N) * len(images_train), cant))
    for i in range(len(images_train)):
        imgGris = color.rgb2gray(images_train[i])
        respBanco = calculateFilterResponse_201719942_201822262(imgGris, filters)
        resp[i*M_N:M_N * (i+1), :] = respBanco

    # TODO Realizar un (1) ciclo que recorra todas las imágenes de entrenamiento
    # TODO En cada iteración:
    #           - Calcular la respuesta de la imagen al banco de filtros (función anterior)
    #           - Almacenar la matriz resultante en el arreglo de respuestas (tenga en cuenta
    #             la posición de los pixeles de cada imagen dentro del arreglo de respuestas)

    # TODO Establecer semilla
    semilla = 0
    # TODO Declarar el modelo de KMeans
    # TODO Ajustar el modelo inicializado al arreglo de resultados del punto anterior
    modelo_kmeans = KMeans(parameters['k'], random_state=semilla).fit(resp)
    # TODO Obtener las coordenadas de los centroides en una variable y almacenarlas
    #       en un diccionario, bajo la llave 'centroids'
    #print(modelo_kmeans.cluster_centers_.shape)
    dic = {'centroids': modelo_kmeans.cluster_centers_}

    # TODO Almacenar el diccionario anterior como un archivo .mat, bajo el nombre
    #       'dictname' (parámetro de entrada)
    savemat(parameters['dict_name'], dic) # PREGUNTAR POR DICTNAME VS DICT_NAME
 # TODO Borrar los comentarios marcados con un TODO.

def CalculateTextonHistogram_201719942_201822262(img_gray, centroids):
    #print(centroids)
    bins = len(centroids)
    """copiaImagen = img_gray.copy()
    for i in range(len(img_gray)):
        for j in range(len(img_gray[0])):
            menorDist = 0
            centroTemp = None
            for k in range(len(centroids[0])):
                pnt1 = np.array([i,j])
                pnt2 = np.array([centroids[0][k], centroids[1][k]])
                if menorDist > distance.euclidean(pnt1, pnt2):
                    menorDist = distance.euclidean(pnt1, pnt2)
                    centroTemp = k
            copiaImagen[i][j] = centroTemp"""

    filtros = loadmat('filterbank.mat')
    respBanco = calculateFilterResponse_201719942_201822262(img_gray, filtros)
    copiaImagen = respBanco.copy()
    #   print(len(respBanco))

    for i in range(len(respBanco)):
        menorDist = 0
        centroTemp = None
        for k in range(len(centroids)):
            distandia_euc = distance.euclidean(respBanco[i], centroids[k]) #np.linalg.norm(respBanco[i]-centroids[k], ord=-1) #SALE ERROR ValueError: autodetected range of [nan, nan] is not finite
            if menorDist > distandia_euc:
                menorDist = distandia_euc
                centroTemp = k
            elif k == 0:
                menorDist = distandia_euc
                centroTemp = k
        #print(i)
        copiaImagen[i] = centroTemp
    #copiaImagen.flatten()
    print(copiaImagen.flatten().shape)
    hist = np.histogram(copiaImagen.flatten(), bins=bins)
    return hist[0]

def main(parameters, perform_train, action):
    if perform_train:
        train(parameters, action = action)
    conf_mat, precision, recall, f_score = validate(parameters, action = action)
    #TODO Imprimir de manera organizada el resumen del experimento.
    # Incluyan los parámetros que usaron y las métricas de validación.
    resp = f"Parámetros:\nNombre diccionario: {parameters['dict_name']:37} | Nombre del modelo: {parameters['name_model']:25} " \
           f"\nNombre del descriptor entrenamiento: {parameters['train_descriptor_name']:15} | Nombre del descriptor validación: {parameters['val_descriptor_name']:15}" \
           f"\nKernel: {parameters['kernel']:51} " \
           f"\nNúmero de clusters: {parameters['k']:40} | Precisión: {precision:44} " \
           f"\nCobertura: {recall:49} | F-score: {f_score:46}\nMatriz de confusión:\n{conf_mat}"
    print(resp)
if __name__ == '__main__':
    """
    Por: Isabela Hernández y Natalia Valderrama
    Este código establece los parámetros de experimentación. Permite extraer
    los descriptores de las imágenes, entrenar un modelo de clasificación con estos
    y validar su desempeño.
    Nota: Este código fue diseñado para los estudiantes de IBIO-3470 2021-10.
    Rogamos no hacer uso de este código por fuera del curso y de este semestre.
    ----------NO OPEN ACCESS!!!!!!!------------
    """
    kernel = "linear" #{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    numero_cluster = 6 #corresponde con el número de clases
    dictName = f'SVMk{kernel}_c{numero_cluster}_dict.mat'
    nombre_modelo = f'SVMk{kernel}_c{numero_cluster}_modelo.npy'#"best_model_E1_201719942_201822262.npy"   #f'{funcion_str}{espacio}_b{numero_bins}_c{numero_cluster}_modelo.npy'
    nombre_entrenamiento = f'SVMk{kernel}_c{numero_cluster}_train.npy'
    nombre_validacion = f'SVMk{kernel}_c{numero_cluster}_val.npy'
    # TODO Establecer los valores de los parámetros con los que van a experimentar.
    # Nota: Tengan en cuenta que estos parámetros cambiarán según los descriptores
    # y clasificadores a utilizar.
    parameters= {'kernel': kernel, 'dict_name': dictName,
             'transform_color_function': color.rgb2gray, # Esto es solo un ejemplo.
             'k': numero_cluster,
             'name_model': nombre_modelo, # No olviden establecer la extensión con la que guardarán sus archivos.
             'train_descriptor_name': nombre_entrenamiento, # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': nombre_validacion} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.
    perform_train = True#True # Cambiar parámetro a False al momento de hacer la entrega
    action = 'save' # Cambiar a None al momento de hacer la entrega
    main(parameters=parameters, perform_train=perform_train, action=action)
# 'bins': numero_bins,

 ##
test = np.array([[0,0,0],[0,0,0],[0,0,0]])
vDeVector = np.transpose(np.array([[1, 2, 3]]))

#test[:,1] = vDeVector[:,0]
#print(test)
miniMatrizParaPamePorqueLeEncanta = np.array([[1, 2, 3], [4, 5, 6]])
test[1:3, :] = miniMatrizParaPamePorqueLeEncanta

a = np.array([1, 2])
b = np.array([2,4])
c = distance.euclidean(a, b)

