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
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from skimage import img_as_float
from data_mp4.functions import JointColorHistogram, CatColorHistogram
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat

def calculate_descriptors(data, parameters):
    if parameters['space'] != 'RGB':
        data = list(map(parameters['transform_color_function'], data))
    bins = [parameters['bins']]*len(data)
    histograms = list(map(parameters['histogram_function'], data, bins))
    # TODO Verificar tamaño de descriptor_matrix igual a # imágenes x dimensión del descriptor
    # se realizan ajustes si son necesarios para cada una de las funciones
    if parameters['histogram_function'] == CatColorHistogram:
        desc_long=parameters['bins']*3 #tamaño descriptor para histogramas contatenados
        descriptor_matrix = np.array(histograms)
    else:
        desc_long=parameters['bins'] ** 3 #tamaño descriptor para histogramas conjuntos
        flat_hists=[]
        for descript in histograms: # ajuste de dimensiones de la matriz de descritptores
            flat_descript=descript.flatten()                             #print(flat_descript.shape)
            flat_hists.append(flat_descript)
        descriptor_matrix = np.array(flat_hists)           #np.array(histograms)                       #print(len(descriptor_matrix),len(descriptor_matrix[0]))
    assert len(descriptor_matrix)*len(descriptor_matrix[0]) == (desc_long)*len(data), 'El tamaño del descriptor no es el adecuado' # verificación
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
        descriptors = calculate_descriptors(images_train, parameters)
        # TODO Guardar matriz de descriptores con el nombre parameters['train_descriptor_name']
        np.save(parameters['train_descriptor_name'], descriptors)         #print(descriptors)
    else:
        y = 0
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['train_descriptor_name']
        # descriptors = np.load(parameters['train_descriptor_name'])
    # TODO Definir una semilla y utilice la misma para todos los experimentos de la entrega.
    semilla = 0
    # TODO Inicializar y entrenar el modelo con los descriptores.
    entrenamiento = skclust.KMeans(parameters['k'], random_state=semilla).fit(descriptors)
    etiquetas = entrenamiento.labels_                 #print(len(etiquetas),len(images_train)) #print(len(data_train),images_train)
    plt.figure()  # se plotean las imágenes resultantes
    for i in range(len(etiquetas)):
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
    plt.show()
    # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
    pickle.dump(entrenamiento, open(parameters['name_model'],'wb'))

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
        descriptors = calculate_descriptors(images_val, parameters)
        if action == 'save':
            # TODO Guardar matriz de descriptores con el nombre parameters['val_descriptor_name']
            np.save(parameters['val_descriptor_name'], descriptors)
    # TODO Cargar el modelo de parameters['name_model']
    modelo = pickle.load(open(parameters['name_model'],'rb'))        #print(type(modelo))
    # TODO Obtener las predicciones para los descriptores de las imágenes de validación
    predicciones = modelo.predict(descriptors)
    anotaciones = []
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
            anotaciones.append(1)#                                              anotaciones.append(mode(diccionario['street']))
    # TODO Obtener las métricas de evaluación
    conf_mat = sk.confusion_matrix(anotaciones, predicciones)
    precision = sk.precision_score(anotaciones, predicciones,average="macro")
    recall = sk.recall_score(anotaciones, predicciones,average="macro")
    f_score = sk.f1_score(anotaciones, predicciones,average="macro")                                                                                                                                #print((2*precision*recall)/(precision+recall))
    return conf_mat, precision, recall, f_score

def main(parameters, perform_train, action):
    if perform_train:
        train(parameters, action = action)
    conf_mat, precision, recall, f_score = validate(parameters, action = action)
    #TODO Imprimir de manera organizada el resumen del experimento.
    # Incluyan los parámetros que usaron y las métricas de validación.
    if parameters['histogram_function'] == CatColorHistogram:
        funcion_str="Concatenados"
    else:
        funcion_str="Conjuntos   "
    resp = f"Parámetros:\nFunción de histograma: {funcion_str:37} | Nombre del modelo: {parameters['name_model']:25} " \
           f"\nNombre del descriptor entrenamiento: {parameters['train_descriptor_name']:15} | Nombre del descriptor validación: {parameters['val_descriptor_name']:15}" \
           f"\nEspacio: {parameters['space']:51} | Número de Bins: {parameters['bins']:39} " \
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
    espacio='Lab' # variables para almacenar parámetros y facilitar su modificación
    hist_function=CatColorHistogram#JointColorHistogram#
    if hist_function == CatColorHistogram:
        funcion_str="Cat_"
    else:
        funcion_str="Join"
    numero_bins = 5
    numero_cluster = 6 #corresponde con el número de clases
    nombre_modelo = "best_model_E1_201719942_201822262.npy"   #f'{funcion_str}{espacio}_b{numero_bins}_c{numero_cluster}_modelo.npy'
    nombre_entrenamiento = f'{funcion_str}{espacio}_b{numero_bins}_c{numero_cluster}_train.npy'
    nombre_validacion = f'{funcion_str}{espacio}_b{numero_bins}_c{numero_cluster}_val.npy'
    # TODO Establecer los valores de los parámetros con los que van a experimentar.
    # Nota: Tengan en cuenta que estos parámetros cambiarán según los descriptores
    # y clasificadores a utilizar.
    parameters= {'histogram_function': hist_function,
             'space': espacio, 'transform_color_function': color.rgb2lab, # Esto es solo un ejemplo.
             'bins': numero_bins, 'k': numero_cluster,
             'name_model': nombre_modelo, # No olviden establecer la extensión con la que guardarán sus archivos.
             'train_descriptor_name': nombre_entrenamiento, # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': nombre_validacion} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.
    perform_train = False#True # Cambiar parámetro a False al momento de hacer la entrega
    action = None#'save' # Cambiar a None al momento de hacer la entrega
    main(parameters=parameters, perform_train=perform_train, action=action)


# TODO Copiar y pegar estas funciones en el script principal (main_Codigo1_Codigo2.py)
# TODO Cambiar el nombre de las funciones para incluir sus códigos de estudiante

def calculateFilterResponse_201719942_201822262(img_gray, filters):
    # TODO Inicializar arreglo de tamaño (MxN) x número de filtros, llamado 'resp'
    resp = np.zeros((len(img_gray)*len(img_gray[0]), len(filters)))
    # TODO Realizar un (1) ciclo que recorra los filtros
    for i in range(len(filters)):
        correlacion = correlate2d(img_gray, filters[i], boundary="symm")
        resp[:, i] = np.transpose(correlacion.flatten())[:,0]
    # TODO En cada iteración:
    #           - Realizar cross-correlación entre la imagen y el filtro. Para ello, utilizar
    #             correlate2d() y los parámetros que considere pertinentes para no perder el
    #             tamaño original de la imagen.
    #           - Convertir el resultado a un vector y almacenarlo en la posición correspondiente
    #             del arreglo inicial.
    return resp


def calculateTextonDictionary_201719942_201822262(images_train, filters, parameters):

    # TODO Inicializar arreglo de respuestas de tamaño [(MxN) x número de imágenes] x número de filtros
    M_N = len(images_train[0]) * len(images_train[0][0])
    resp = np.zeros(((M_N) * len(images_train), len(filters)))
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
    dic = {'centroids': modelo_kmeans.cluster_centers_}
    # TODO Almacenar el diccionario anterior como un archivo .mat, bajo el nombre
    #       'dictname' (parámetro de entrada)
    savemat(parameters['dictname'], dic) # PREGUNTAR POR DICTNAME VS DICT_NAME
 # TODO Borrar los comentarios marcados con un TODO.

def CalculateTextonHistogram_201719942_201822262(img_gray, centroids):
    bins = len(centroids)
    dic = {}
    for i in range(len(img_gray)):
        for j in range(len(img_gray[0])):
            menorDist = 0
            centroTemp = None
            for k in range(len(centroids[0])):
                
    return hist
 ##
test = np.array([[0,0,0],[0,0,0],[0,0,0]])
vDeVector = np.transpose(np.array([[1, 2, 3]]))
print(test)
print(vDeVector)
#test[:,1] = vDeVector[:,0]
#print(test)
miniMatrizParaPamePorqueLeEncanta = np.array([[1, 2, 3], [4, 5, 6]])
test[1:3, :] = miniMatrizParaPamePorqueLeEncanta
print(test)


