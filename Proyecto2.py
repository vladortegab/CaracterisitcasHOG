#os.getcwd()
#RutaProyecto = os.getcwd()
#os.listdir(RutaProyecto + '\\BASEDATOS')

#a = ['FRESA','MANGO','MANZANA','PERA']
#fruits_dict = {fruit: i for i, fruit in enumerate(a)}
#clases_dict[clase]

#tenemos 4 carpetas fresa, mango, manzana, pera, y en cada una tienes un banco de imagenes
import os, sys;
import numpy as np;
import cv2 as cv;

#llamamos a esas carpetas desde una carpeta raiz que las contiene
RutaProyecto = os.getcwd();
Clases = os.listdir(RutaProyecto + '\\RESIZE_IMAGENES_SALIDA');
clases_dict = {itemClass: i for i, itemClass in enumerate(Clases)} # esto ya es generico, frutas, sillas, etc

hog = cv.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

img = np.zeros((64, 64, 1), dtype=np.uint8)
desc= hog.compute(img)
VectorTraining=desc.T
YOut = []
YOut.append(0)

for clase in Clases:
    print(clase)
    Path = RutaProyecto + '\\RESIZE_IMAGENES_SALIDA\\' + clase
    ListaImagenes = os.listdir(Path)
    for imagen in ListaImagenes:                       
        imagenOriginal = cv.imread(Path + '\\' + imagen, 0)
        if(imagenOriginal is None or img.shape == (0, 0)):
            print("Error:" + Path + '\\' + imagen)
        else:
            
            imagenResize = cv.resize(imagenOriginal, (64,64), interpolation = cv.INTER_AREA)
            descriptor = hog.compute(imagenResize)
            #VectorTraining = np.append(VectorTraining, descriptor, axis=0)
            VectorTraining = np.vstack([VectorTraining, descriptor.T])
            YOut.append(clases_dict[clase])
   
#VectorTraining.pop(0)            
VectorTraining = np.delete(VectorTraining, 0, axis = 0) #ESto pasa de 334 a 333
#del VectorTraining[0]            
YOut.pop(0)   
         
 #llamamos a la carpetas salida que tendra el archivo vectorTraining y VectorY
rutaSalida = RutaProyecto + '\\BASEDATOS_SALIDA'   
if(os.path.exists(rutaSalida)):
    print('Guardando el archivo')
    np.save(rutaSalida + '\\VectorTraining',VectorTraining)
    np.save(rutaSalida + '\\vectorY',YOut)
else:
    print('No existe, se crea la carpeta')
    os.mkdir(rutaSalida)
    np.save(rutaSalida + '\\VectorTraining',VectorTraining);
    np.save(rutaSalida + '\\vectorY',YOut)
        
        
#Aqui tenemos el primer pedazo del proyecto, ahora falta al entrenamiento        