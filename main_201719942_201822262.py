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

def calculate_descriptors(data, parameters):
    if parameters['space'] != 'RGB':
        data = list(map(parameters['transform_color_function'], data))
    bins = [parameters['bins']]*len(data)
    histograms = list(map(parameters['histogram_function'], data, bins))
    # TODO Verificar tamaño de descriptor_matrix igual a # imágenes x dimensión del descriptor
    flat_hists=[]
    for descript in histograms:
        flat_descript=descript.flatten()
        #print(flat_descript.shape)
        flat_hists.append(flat_descript)
    descriptor_matrix = np.array(flat_hists)#np.array(histograms)
    assert len(descriptor_matrix)*len(descriptor_matrix[0]) == (parameters['bins']**3)*len(data), 'El tamaño del descriptor no es el adecuado'
    return descriptor_matrix

diccionario = {}
def train(parameters, action):
    data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')
    #images_train = list(map(io.imread, glob.glob(data_train)))
    rutas = glob.glob(data_train)
    images_train = list(map(io.imread, rutas))
    nombres = []
    global diccionario
    for i in rutas:
        ruta = i.split(i[8])
        #print( ruta[-1])
        nombres.append(ruta[-1].split('_')[0])
    if action == 'save':
        descriptors = calculate_descriptors(images_train, parameters)
        # TODO Guardar matriz de descriptores con el nombre parameters['train_descriptor_name']
        #print(descriptors)
        np.save(parameters['train_descriptor_name'], descriptors)
    else:
        y = 0
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['train_descriptor_name']
        # descriptors = np.load(parameters['train_descriptor_name'])
    # TODO Definir una semilla y utilice la misma para todos los experimentos de la entrega.
    semilla = 0
    # TODO Inicializar y entrenar el modelo con los descriptores.
    entrenamiento = skclust.KMeans(parameters['k'], random_state=semilla).fit(descriptors)
    etiquetas = entrenamiento.labels_
    #print(len(etiquetas),len(images_train))
    #print(len(data_train),images_train)
    plt.figure()  # se plotean las imágenes resultantes
    for i in range(len(etiquetas)):
        if nombres[i] not in diccionario:
            diccionario[nombres[i]] = [etiquetas[i]]
        else:
            diccionario[nombres[i]].append(etiquetas[i])
            #diccionario[data_train[i]] = etiquetas[i]
        plt.subplot(8,6,i+1)
        plt.title(f"{nombres[i]}\nCluster{etiquetas[i]}")
        plt.axis("off")
        plt.imshow(images_train[i])
    print(diccionario)
    plt.show()
    # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
    pickle.dump(entrenamiento, open(parameters['name_model'],'wb'))
    #np.save(parameters['name_model'], entrenamiento)

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
    modelo = pickle.load(open(parameters['name_model'],'rb'))
    #modelo = np.load(parameters['name_model'], allow_pickle=True)
    #print(type(modelo))
    # TODO Obtener las predicciones para los descriptores de las imágenes de validación
    predicciones = modelo.predict(descriptors)
    anotaciones = []
    for i in nombres:
        if i == "glacier":
            anotaciones.append(mode(diccionario['glacier']))
        elif i == "buildings":
            anotaciones.append(mode(diccionario['buildings']))
        elif i == "forest":
            anotaciones.append(mode(diccionario['forest']))
        elif i == "mountains":
            anotaciones.append(mode(diccionario['mountains']))
        elif i == "sea":
            anotaciones.append(mode(diccionario['sea']))
        else:
            anotaciones.append(mode(diccionario['street']))
    print(anotaciones)
    # TODO Obtener las métricas de evaluación
    conf_mat = sk.confusion_matrix(anotaciones, predicciones)
    precision = sk.precision_score(anotaciones, predicciones)
    #print(precision)
    recall = sk.recall_score(anotaciones, predicciones)
    f_score = sk.f1_score(anotaciones, predicciones)
    return conf_mat, precision, recall, f_score

def main(parameters, perform_train, action):
    if perform_train:
        train(parameters, action = action)
    conf_mat, precision, recall, f_score = validate(parameters, action = action)
    #TODO Imprimir de manera organizada el resumen del experimento.
    # Incluyan los parámetros que usaron y las métricas de validación.
    resp = f"Parámetros:\nEspacio: {parameters['space']}\nNúmero de Bins: {parameters['bins']}\nNúmero de clusters: " \
           f"{parameters['k']}\nNombre del modelo: {parameters['name_model']}\nNombre del descriptor entrenamiento:" \
           f"{parameters['train_descriptor_name']}\nNombre del descriptor validación: {parameters['val_descriptor_name']}" \
           f"\nMatriz de confusión: {conf_mat}\nPrecisión: {precision}\nCobertura: {recall}\nF-score: {f_score}"
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
    numero_bins = 20
    numero_cluster = 6 #corresponde con el número de clases
    nombre_modelo = 'modelo1.npy'
    nombre_entrenamiento = 'entrenamiento1.npy'
    nombre_validacion = 'validacion1.npy'
    # TODO Establecer los valores de los parámetros con los que van a experimentar.
    # Nota: Tengan en cuenta que estos parámetros cambiarán según los descriptores
    # y clasificadores a utilizar.
    parameters= {'histogram_function': JointColorHistogram,
             'space': 'Lab', 'transform_color_function': color.rgb2lab, # Esto es solo un ejemplo.
             'bins': numero_bins, 'k': numero_cluster,
             'name_model': nombre_modelo, # No olviden establecer la extensión con la que guardarán sus archivos.
             'train_descriptor_name': nombre_entrenamiento, # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': nombre_validacion} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

    perform_train = True # Cambiar parámetro a False al momento de hacer la entrega
    action = 'save' # Cambiar a None al momento de hacer la entrega
    main(parameters=parameters, perform_train=perform_train, action=action)


