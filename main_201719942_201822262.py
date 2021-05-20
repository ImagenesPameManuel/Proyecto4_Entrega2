#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto4 Entrega2
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
import os
import glob
import pickle
from functools import partial
import numpy as np
import sklearn.cluster as skclust
import sklearn.metrics as sk
import tqdm as tqdm
from skimage import io
from skimage import color
from sklearn import ensemble
from skimage.feature import hog
from data_mp4.functions import JointColorHistogram, CatColorHistogram
from data_mp4.pykernels.regular import GeneralizedHistogramIntersection
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat
from scipy.spatial import distance
from sklearn import svm
from tqdm import tqdm
def calculateFilterResponse_201719942_201822262(img_gray, filters):
    cant = filters['filterbank'].shape[2]
    resp = np.zeros((len(img_gray)*len(img_gray[0]), cant))
    for i in tqdm(range(filters['filterbank'].shape[2])):
        correlacion = correlate2d(img_gray, filters['filterbank'][:, :, i], boundary="symm", mode='same')
        correlacion = np.transpose(np.array([correlacion.flatten()]))
        resp[:, i] = (correlacion)[:, 0]
    return resp
def calculateTextonDictionary_201719942_201822262(images_train, filters, parameters):
    cant = filters['filterbank'].shape[2]
    M_N = len(images_train[0]) * len(images_train[0][0])
    resp = np.zeros(((M_N) * len(images_train), cant))
    for i in range(len(images_train)):
        imgGris = color.rgb2gray(images_train[i])
        respBanco = calculateFilterResponse_201719942_201822262(imgGris, filters)
        resp[i*M_N:M_N * (i+1), :] = respBanco
    semilla = 0
    modelo_kmeans = KMeans(parameters['k'], random_state=semilla).fit(resp)
    dic = {'centroids': modelo_kmeans.cluster_centers_}
    savemat(parameters['dict_name'], dic)
def CalculateTextonHistogram_201719942_201822262(img_gray, centroids):
    bins = len(centroids)
    filtros = loadmat('filterbank.mat')
    respBanco = calculateFilterResponse_201719942_201822262(img_gray, filtros)
    copiaImagen = respBanco.copy()
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
        copiaImagen[i] = centroTemp
    hist = np.histogram(copiaImagen.flatten(), bins=bins)
    return hist[0]
def calculate_descriptors(data, parameters,calculate_dict=True):
    if parameters['descriptores']=="Color":
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
    elif parameters['descriptores']=="Textones":
        filtros = loadmat('filterbank.mat')
        dataGris = list(map(parameters['transform_color_function'], data))
        if calculate_dict:
            calculateTextonDictionary_201719942_201822262(data, filtros, parameters)
        dic = loadmat(parameters['dict_name'])
        centroides = dic['centroids']
        mapfunc = partial(CalculateTextonHistogram_201719942_201822262, centroids=centroides)
        descriptor_matrix = list(map(mapfunc, dataGris))
        assert len(descriptor_matrix) * len(descriptor_matrix[0]) == (
                    len(data) * parameters['k']), 'El tamaño del descriptor no es el adecuado'  # verificación
    elif parameters['descriptores'] == "HOG":
        descriptor_matrix = list( map(partial(hog, block_norm=parameters['norm_block'], pixels_per_cell=parameters['pixels_per_cell2']),                data))
    return descriptor_matrix
diccionario = {} # variable global para guardar cluster asignado para cada una de las clases trabajadas
def train(parameters, action):
    data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')  # carga imágenes train
    rutas = glob.glob(data_train)
    images_train = list(map(io.imread, rutas))
    nombres = []
    global diccionario  # variable global para conteo de asignación de clusters para cada clase
    for i in rutas:
        ruta = i.split(i[8])  # extracción de anotación            #print( ruta[-1])
        nombres.append(ruta[-1].split('_')[0])
    if action == 'save':
        descriptors = calculate_descriptors(images_train, parameters, False)
        # TODO Guardar matriz de descriptores con el nombre parameters['train_descriptor_name']
        np.save(parameters['train_descriptor_name'], descriptors)  # print(descriptors)
    else:
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['train_descriptor_name']
        descriptors = np.load(parameters['train_descriptor_name'], 'rb')
    # TODO Definir una semilla y utilice la misma para todos los experimentos de la entrega.
    if parameters["metodo"]=="SVM":
        kernel = parameters['kernel']
        # TODO Inicializar y entrenar el modelo con los descriptores.
        entrenamiento_SVM = svm.SVC(kernel=kernel).fit(descriptors, nombres)
        # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
        pickle.dump(entrenamiento_SVM, open(parameters['name_model'], 'wb'))
    elif parameters["metodo"]=="Kmeans":
        semilla = 0
        # TODO Inicializar y entrenar el modelo con los descriptores.
        entrenamiento = skclust.KMeans(parameters['k'], random_state=semilla).fit(descriptors)
        etiquetas = entrenamiento.labels_  # print(len(etiquetas),len(images_train)) #print(len(data_train),images_train)
        for i in range(len(etiquetas)):
            if nombres[i] not in diccionario:  # conteo de asignación de clusters para cada una de las etiquetas
                diccionario[nombres[i]] = {etiquetas[i]: 1}
            else:
                if etiquetas[i] not in diccionario[nombres[i]]:
                    diccionario[nombres[i]][etiquetas[i]] = 1
                else:
                    diccionario[nombres[i]][etiquetas[i]] += 1
        # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
        pickle.dump(entrenamiento, open(parameters['name_model'], 'wb'))
    elif parameters["metodo"] == "RF":
        randm_state = 0
        # TODO Inicializar y entrenar el modelo con los descriptores.
        pre_entrenamiento_RF = ensemble.RandomForestClassifier(random_state=randm_state,  n_estimators=parameters['n_estimators'],                max_depth=parameters['max_depth'])
        entrenamiento_RF = pre_entrenamiento_RF.fit(descriptors, nombres)
        # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
        pickle.dump(entrenamiento_RF, open(parameters['name_model'], 'wb'))
def validate(parameters, action):
    data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
    rutas = glob.glob(data_val)
    images_val = list(map(io.imread, rutas))
    nombres = []
    for i in rutas:
        ruta = i.split(i[8])
        nombres.append(ruta[-1].split('_')[0])
    if action == 'load':
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['val_descriptor_name']
        descriptors = np.load(parameters['val_descriptor_name'])
    else:
        descriptors = calculate_descriptors(images_val, parameters, False)
        if action == 'save':
            # TODO Guardar matriz de descriptores con el nombre parameters['val_descriptor_name']
            np.save(parameters['val_descriptor_name'], descriptors)
    # TODO Cargar el modelo de parameters['name_model']
    modelo = pickle.load(open(parameters['name_model'], 'rb'))  # print(type(modelo))
    # TODO Obtener las predicciones para los descriptores de las imágenes de validación
    predicciones = modelo.predict(descriptors)
    # TODO Obtener las métricas de evaluación
    conf_mat = sk.confusion_matrix(nombres, predicciones)
    precision = sk.precision_score(nombres, predicciones, average="macro", zero_division=1)
    recall = sk.recall_score(nombres, predicciones, average="macro", zero_division=1)
    f_score = sk.f1_score(nombres, predicciones, average="macro")  # print((2*precision*recall)/(precision+recall))
    return conf_mat, precision, recall, f_score
def main(parameters, perform_train, action):
    if perform_train:
        train(parameters, action=action)
    conf_mat, precision, recall, f_score = validate(parameters, action=action)
    # TODO Imprimir de manera organizada el resumen del experimento.
    # Incluyan los parámetros que usaron y las métricas de validación.
    resp = f"Parámetros:\nNombre diccionario: {parameters['dict_name']:37} | Nombre del modelo: {parameters['name_model']:25} " \
           f"\nNombre del descriptor entrenamiento: {parameters['train_descriptor_name']:15} | Nombre del descriptor validación: {parameters['val_descriptor_name']:15}" \
           f"\nKernel: {parameters['kernel']} " \
           f"\nTamaño del diccionario: {parameters['k']:40} | Precisión: {precision:44} " \
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
    kernel = GeneralizedHistogramIntersection() #{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    numero_cluster = 6 #corresponde con el número de clases
    dictName = 'SVMklinear_c6_dict.mat' #f'SVMk{kernel}_c{numero_cluster}_dict.mat'
    nombre_modelo = f'SVMkGenHist_c{numero_cluster}_modelo.npy'#f'SVMk{kernel}_c{numero_cluster}_modelo.npy'#"best_model_E1_201719942_201822262.npy"   #f'{funcion_str}{espacio}_b{numero_bins}_c{numero_cluster}_modelo.npy'
    nombre_entrenamiento = f'SVMkGenHist_c{numero_cluster}_train.npy'
    nombre_validacion = f'SVMkGenHist_c{numero_cluster}_val.npy'
    # TODO Establecer los valores de los parámetros con los que van a experimentar.
    # Nota: Tengan en cuenta que estos parámetros cambiarán según los descriptores
    # y clasificadores a utilizar.
    parameters= {"descriptores":"Textones","metodo":"SVM",'kernel': kernel, 'dict_name': dictName,
             'transform_color_function': color.rgb2gray, # Esto es solo un ejemplo.
             'k': numero_cluster,
             'name_model': nombre_modelo, # No olviden establecer la extensión con la que guardarán sus archivos.
             'train_descriptor_name': nombre_entrenamiento, # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': nombre_validacion} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.
    perform_train = True#True # Cambiar parámetro a False al momento de hacer la entrega
    action = "save" # Cambiar a None al momento de hacer la entrega
    main(parameters=parameters, perform_train=perform_train, action=action)

