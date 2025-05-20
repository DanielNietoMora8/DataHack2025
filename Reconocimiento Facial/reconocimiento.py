import cv2
import os
import imutils

# Ruta de datos y personas
dataPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/Data'
imagePaths = os.listdir(dataPath)
print('Personas detectadas en el modelo:', imagePaths)

# Cargar modelo previamente entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# Cargar video
videoPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/VideoPrueba/AlvaroMorte.mp4'
cap = cv2.VideoCapture(videoPath)

# Clasificador Haar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if faceClassif.empty():
    print("❌ No se pudo cargar el clasificador Haar.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Fin del video o error al leer.")
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        label, confidence = face_recognizer.predict(rostro)

        # Mostrar resultados
        if confidence < 70:  # Umbral ajustable
            nombre = imagePaths[label]
            color = (0, 255, 0)
        else:
            nombre = 'Desconocido'
            color = (0, 0, 255)

        cv2.putText(frame, f'{nombre} ({round(confidence, 2)})', (x, y - 10), 2, 0.8, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Reconocimiento Facial en Video', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
