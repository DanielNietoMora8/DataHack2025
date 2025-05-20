import cv2
import os
import numpy as np

dataPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print(f'Leyendo imágenes de {nameDir}...')
    count = 0

    for fileName in os.listdir(personPath):
        imgPath = os.path.join(personPath, fileName)
        img = cv2.imread(imgPath, 0)  # Leer en escala de grises
        if img is None:
            print(f'Error al cargar la imagen: {imgPath}')
            continue

        # Preprocesamiento: equalizar histograma para mejorar contraste
        img = cv2.equalizeHist(img)

        facesData.append(img)
        labels.append(label)
        count += 1

    print(f'Imágenes procesadas para {nameDir}: {count}')
    label += 1

labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=123.0)

print("Entrenando el reconocedor de rostros...")
face_recognizer.train(facesData, labels)

face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado correctamente.")
