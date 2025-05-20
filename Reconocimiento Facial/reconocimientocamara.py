import cv2
import os
import imutils

# Ruta donde están las carpetas con los nombres de las personas
dataPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/Data'
imagePaths = os.listdir(dataPath)
print('Personas detectadas en el modelo:', imagePaths)

# Cargar el modelo LBPH previamente entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

# Activar la cámara (usa 0 o 1 según tu sistema)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usa CAP_DSHOW para evitar advertencias en Windows

# Cargar el clasificador Haar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if faceClassif.empty():
    print("❌ No se pudo cargar el clasificador Haar.")
    exit()

print("🎥 Iniciando cámara. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo acceder a la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        label, confidence = face_recognizer.predict(rostro)

        if confidence < 70:  # Ajusta este umbral si lo necesitas
            nombre = imagePaths[label]
            color = (0, 255, 0)
        else:
            nombre = 'Desconocido'
            color = (0, 0, 255)

        cv2.putText(frame, f'{nombre} ({round(confidence, 2)})', (x, y - 10), 2, 0.8, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Reconocimiento Facial en Vivo', frame)

    if cv2.waitKey(1) == 27:  # ESC para salir
        print("🛑 Reconocimiento detenido por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()
